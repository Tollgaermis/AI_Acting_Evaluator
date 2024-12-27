# import whisper
# import librosa
# import numpy as np

# # Load Whisper model
# model = whisper.load_model("base")

# # Step 1: Transcribe audio and get word timestamps
# def transcribe_audio(audio_path):
#     """
#     Transcribe an audio file and return a list of words with their timestamps.

#     This function uses the Whisper model to transcribe the given audio file. It then extracts 
#     the transcribed words along with their start and end timestamps and stores them in a list 
#     of dictionaries. Each dictionary contains the word text, start time, and end time for each word.

#     Args:
#         audio_path (str): Path to the audio file to be transcribed. The file should be in a format 
#                           supported by the Whisper model (e.g., MP3, WAV).

#     Returns:
#         List[dict]: A list of dictionaries, where each dictionary contains the following keys:
#             - "text" (str): The transcribed word.
#             - "start" (float): The start time of the word in seconds.
#             - "end" (float): The end time of the word in seconds.

#     Example:
#         >>> words = transcribe_audio("path/to/audio.mp3")
#         >>> print(words)
#         [{'text': 'Hello', 'start': 0.0, 'end': 0.5}, {'text': 'world', 'start': 0.5, 'end': 1.0}]
#     """
#     result = model.transcribe(audio_path, word_timestamps=True)
#     words_with_timestamps = []
#     for segment in result["segments"]:
#         for word in segment["words"]:
#             words_with_timestamps.append({
#                 "text": word["word"],
#                 "start": word["start"],
#                 "end": word["end"]
#             })
#     return words_with_timestamps

# # Step 2: Extract features using Librosa
# def extract_audio_features(audio_path, start_time, end_time, sr=16000):
#     """
#     Extract pitch and intensity features from a specific segment of audio.

#     This function uses Librosa to extract the pitch (using the PYIN algorithm) and intensity (root mean square) 
#     features from a segment of audio defined by the given start and end times.

#     Args:
#         audio_path (str): Path to the audio file to extract features from.
#         start_time (float): Start time (in seconds) of the audio segment.
#         end_time (float): End time (in seconds) of the audio segment.
#         sr (int, optional): Sampling rate for the audio, default is 16000 Hz.

#     Returns:
#         tuple: A tuple containing:
#             - pitch (numpy.ndarray): The pitch values for the segment, extracted using PYIN.
#             - intensity (numpy.ndarray): The root mean square intensity for the segment.

#     Example:
#         >>> pitch, intensity = extract_audio_features("audio.mp3", 0.0, 5.0)
#         >>> print(pitch, intensity)
#     """
#     y, _ = librosa.load(audio_path, sr=sr, offset=start_time, duration=end_time - start_time)
#     pitch = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)[0]
#     intensity = librosa.feature.rms(y=y)
#     return pitch, intensity

# # Step 3: Detect emphasis based on feature changes
# def detect_emphasis(audio_path, words_with_timestamps, n):
#     """
#     Detect the most emphasized words based on intensity and pitch features.

#     This function processes the audio segment for each word, extracts its pitch and intensity features, 
#     and determines the most emphasized words by comparing their mean intensity. 
#     The top `n` emphasized words are returned, with `n` being a user-defined parameter.

#     Args:
#         audio_path (str): Path to the audio file to analyze.
#         words_with_timestamps (list of dict): A list of dictionaries containing word text and timestamps, 
#                                               as returned by the `transcribe_audio` function.
#         n (int): The number of most emphasized words to return. If `n` exceeds the total number of words, 
#                  all words are returned.

#     Returns:
#         list: A list of the `n` most emphasized words based on their intensity.

#     Example:
#         >>> words = detect_emphasis("audio.mp3", words_with_timestamps, 3)
#         >>> print(words)
#         ['hello', 'world', 'great']
#     """
#     emphasized_words = []

#     for word_info in words_with_timestamps:
#         word = word_info["text"]
#         start_time = word_info["start"]
#         end_time = word_info["end"]

#         # Extract pitch and intensity for the word
#         pitch, intensity = extract_audio_features(audio_path, start_time, end_time)
#         mean_intensity = np.mean(intensity)

#         emphasized_words.append({
#             "word": word,
#             "intensity": mean_intensity
#         })

#     # Sort words by intensity (descending) and select the top n words
#     emphasized_words_sorted = sorted(emphasized_words, key=lambda x: x["intensity"], reverse=True)

#     # If n is larger than the total number of words, return all words
#     n = min(n, len(emphasized_words_sorted))

#     # Get the n most emphasized words
#     most_emphasized = [word_info["word"] for word_info in emphasized_words_sorted[:n]]

#     return most_emphasized


# # Step 4: Main function to detect emphasis
# def main(audio_path, n=1):
#     print("Transcribing audio...")
#     words_with_timestamps = transcribe_audio(audio_path)
#     print(f"Transcription complete. Words: {[w['text'] for w in words_with_timestamps]}")

#     emphasized_words = detect_emphasis(audio_path, words_with_timestamps, n)
#     print(f"The {n} most emphasized words are: {emphasized_words}")

# # Input audio file
# audio_path = "kanka.mp3"  # Replace with your file path

# # Example usage
# n = 2  # Set the number of emphasized words to find (can be set dynamically)
# main(audio_path, n)

import whisper
import librosa
import numpy as np

# Load Whisper model globally
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    """
    Transcribe an audio file and return a list of words with their timestamps.

    Args:
        audio_path (str): Path to the audio file to be transcribed.

    Returns:
        List[dict]: A list of dictionaries with "text", "start", and "end" keys.
    """
    try:
        result = model.transcribe(audio_path, word_timestamps=True)
        words_with_timestamps = [
            {
                "text": word["word"],
                "start": word["start"],
                "end": word["end"]
            }
            for segment in result["segments"]
            for word in segment["words"]
        ]
        return words_with_timestamps
    except Exception as e:
        raise ValueError(f"Error during transcription: {e}")

def extract_audio_features(audio_path, start_time, end_time, sr=16000):
    """
    Extract pitch and intensity features from a specific segment of audio.

    Args:
        audio_path (str): Path to the audio file.
        start_time (float): Start time of the audio segment.
        end_time (float): End time of the audio segment.
        sr (int): Sampling rate.

    Returns:
        tuple: Pitch (numpy.ndarray) and intensity (numpy.ndarray).
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, offset=start_time, duration=end_time - start_time)
        pitch = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)[0]
        intensity = librosa.feature.rms(y=y)
        return pitch, intensity
    except Exception as e:
        raise ValueError(f"Error extracting audio features: {e}")

def detect_emphasis(audio_path, words_with_timestamps, n):
    """
    Detect the most emphasized words based on intensity features.

    Args:
        audio_path (str): Path to the audio file.
        words_with_timestamps (list): List of word dictionaries from transcription.
        n (int): Number of top emphasized words to return.

    Returns:
        List[str]: List of emphasized words sorted by intensity.
    """
    try:
        emphasized_words = []

        for word_info in words_with_timestamps:
            word = word_info["text"]
            start_time = word_info["start"]
            end_time = word_info["end"]

            # Extract features
            _, intensity = extract_audio_features(audio_path, start_time, end_time)
            mean_intensity = np.mean(intensity)

            emphasized_words.append({
                "word": word,
                "intensity": mean_intensity
            })

        # Sort by intensity and get the top n words
        emphasized_words_sorted = sorted(emphasized_words, key=lambda x: x["intensity"], reverse=True)
        most_emphasized = [entry["word"] for entry in emphasized_words_sorted[:n]]
        return most_emphasized
    except Exception as e:
        raise ValueError(f"Error detecting emphasis: {e}")
