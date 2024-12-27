import numpy as np
import torch
import librosa
import os
from scipy.io.wavfile import write
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from whisper import load_model as load_stt_model  # For speech-to-text
import torch.nn as nn

# Define PAD values for emotions
emotion_pad_values = {
    "Happy": [2.77, 1.21, 1.42],
    "Boring": [-0.53, -1.25, -0.84],
    "Sad": [-0.89, 0.17, -0.70],
    "Fear": [-0.93, 1.30, -0.64],
    "Anxiety": [-0.95, 0.32, -0.63],
    "Disgust": [-1.80, 0.40, 0.67],
    "Anger": [-2.08, 1.00, 1.12],
    "Neutral": [0.00, 0.00, 0.00]
}


# Classify emotion based on PAD values
def classify_emotion(predicted_pad):
    min_distance = float("inf")
    closest_emotion = None
    for emotion, pad_values in emotion_pad_values.items():
        distance = np.sqrt(sum((predicted_pad[i] - pad_values[i]) ** 2 for i in range(3)))
        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion
    return closest_emotion

# Regression Head for emotion prediction
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 3)  # Output: [arousal, dominance, valence]

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        k = 1/x
        h = 6/k
        x = -3 + h
        return x

# Emotion Model
class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = torch.mean(hidden_states, dim=1)  # Mean pooling over time dimension
        logits = self.classifier(hidden_states)
        return logits

# Initialize model and processor
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

# Set device
device = torch.device("cpu")
model.to(device)
model.eval()

# Whisper STT model
stt_model = load_stt_model("base")

# Emotion detection function
def detect_emotion(audio_input, sample_rate):
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(inputs["input_values"])
        arousal, dominance, valence = logits[0].cpu().numpy()

    # Convert PAD to emotion classification
    emotion = classify_emotion([valence, arousal, dominance])

    return {
        "Pleasure": float(round(valence, 3)),
        "Arousal": float(round(arousal, 3)),
        "Dominance": float(round(dominance, 3)),
        "Emotion": emotion,
    }

# Transcribe audio with timestamps
def transcribe_audio_with_timestamps(audio_path):
    result = stt_model.transcribe(audio_path, word_timestamps=True)
    return result["text"], result["segments"]

# Split audio based on the word "But"
def split_audio_on_word(audio_path, output_dir="output", word=" But"):
    try:
        transcription, word_timestamps = transcribe_audio_with_timestamps(audio_path)
        print(f"Transcription: {transcription}")

        # Find the timestamp for the specified word
        split_word_time = None
        for segment in word_timestamps:
            for word_entry in segment["words"]:
                if word_entry["word"] == word:
                    split_word_time = word_entry["start"]
                    break
            if split_word_time:
                break

        if split_word_time is None:
            raise ValueError(f"The word '{word}' was not found in the transcription.")

        # Split audio
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        segment1 = audio[: int(split_word_time * sample_rate)]
        segment2 = audio[int(split_word_time * sample_rate):]

        # Save the segments
        os.makedirs(output_dir, exist_ok=True)
        write(f"{output_dir}/segment1.wav", sample_rate, segment1)
        write(f"{output_dir}/segment2.wav", sample_rate, segment2)

        # Perform emotion detection on each segment
        emotion1 = detect_emotion(segment1, sample_rate)
        emotion2 = detect_emotion(segment2, sample_rate)

        return {
            "Segment 1 Emotion": emotion1,
            "Segment 2 Emotion": emotion2,
            "Split Word": word,
            "Split Time (seconds)": split_word_time,
            "Transcription": transcription,
        }

    except Exception as e:
        print(f"Error in split_audio_on_word: {e}")
        raise
