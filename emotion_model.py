import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

# Define PAD values for each emotion
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

# Classify emotion based on closest PAD values
def classify_emotion(predicted_pad):
    min_distance = float("inf")
    closest_emotion = None
    for emotion, pad_values in emotion_pad_values.items():
        # Calculate Euclidean distance
        distance = np.sqrt(sum((predicted_pad[i] - pad_values[i]) ** 2 for i in range(3)))
        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion
    return closest_emotion

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 3)  # Output 3 values: arousal, dominance, valence

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

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return logits  # Output is [arousal, dominance, valence]

# Initialize model and processor globally for reuse
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to predict emotion for given audio
def predict_emotion_continuous(audio_input):
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model(inputs['input_values'])
        arousal, dominance, valence = output[0].cpu().numpy()

    pleasure = valence  # Map model's valence to PAD's pleasure
    emotion = classify_emotion([pleasure, arousal, dominance])

    return {
        "pleasure": round(pleasure, 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
        "emotion": emotion
}

