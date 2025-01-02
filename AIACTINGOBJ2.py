import numpy as np
import torch
import librosa
import re
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

def split_audio_on_word(audio_path, output_dir="output", word=None):
    """
    Split the audio file into two segments based on the first occurrence of a specified shifting word.
    Args:
        audio_path (str): Path to the recorded audio file.
        output_dir (str): Directory to save split audio segments.
        word (str): The shifting word (from the random sentence).
    Returns:
        dict: Contains emotions for both segments, transcription, and split metadata.
    """
    if not word:
        raise ValueError("A shifting word must be provided for splitting.")

    
    try:
        # Transcribe the audio and get timestamps
        transcription, word_timestamps = transcribe_audio_with_timestamps(audio_path)
        print(f"Transcription: {transcription}")

        # Clean the transcription (remove punctuation, lowercase)
        cleaned_transcription = re.sub(r'[^\w\s]', '', transcription).lower()
        
        # Find the timestamp for the specified shifting word
        split_word_time = None
        for segment in word_timestamps:
            for word_entry in segment["words"]:
                # Clean each word entry (remove punctuation, lowercase)
                cleaned_word = re.sub(r'[^\w\s]', '', word_entry["word"]).lower()

                if cleaned_word == word:  # Case-insensitive match after cleaning
                    split_word_time = word_entry["start"]
                    break
            if split_word_time:
                break

        if split_word_time is None:
            raise ValueError(f"The word '{word}' was not found in the transcription.")

        # Split the audio into two segments based on the timestamp
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        segment1 = audio[: int(split_word_time * sample_rate)]
        segment2 = audio[int(split_word_time * sample_rate):]

        # Save the audio segments
        os.makedirs(output_dir, exist_ok=True)
        write(f"{output_dir}/segment1.wav", sample_rate, segment1)
        write(f"{output_dir}/segment2.wav", sample_rate, segment2)

        # Perform emotion detection on each segment
        emotion1 = detect_emotion(segment1, sample_rate)
        emotion2 = detect_emotion(segment2, sample_rate)

        # Return the results as a dictionary
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
