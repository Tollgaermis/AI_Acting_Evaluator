from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
import numpy as np
import random
from pydub import AudioSegment
import os
import torch
import librosa
from transformers import Wav2Vec2Processor
from emotion_model import EmotionModel, classify_emotion  # Import your existing model and logic
from emphasis import transcribe_audio, detect_emphasis  # Import emphasis functions
from werkzeug.utils import secure_filename
from AIACTINGOBJ2 import (
    transcribe_audio_with_timestamps,
    detect_emotion,
    split_audio_on_word,
)

# Initialize Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Tells flask-sqlalchemy what database to connect to
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
# Enter a secret key
app.config["SECRET_KEY"] = "ENTER YOUR SECRET KEY"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB, adjust as needed

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

# Initialize flask-sqlalchemy extension
db = SQLAlchemy()
 
# LoginManager is needed for our application 
# to be able to log in and out users
login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = "login"

# Create user model
class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True,
                         nullable=False)
    password = db.Column(db.String(250),
                         nullable=False)

class EmphasisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    audio_file_name = db.Column(db.String(250), nullable=False)
    emphasized_words = db.Column(db.String(500), nullable=False)  # Store as comma-separated values or JSON
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    num_words = db.Column(db.Integer, nullable=False)

    user = db.relationship('Users', backref=db.backref('emphasis_results', lazy=True))

class EmotionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    audio_file_name = db.Column(db.String(250), nullable=False)
    emotion = db.Column(db.String(500), nullable=False)  # Store as comma-separated values or JSON
    pleasure = db.Column(db.Float)
    arousal = db.Column(db.Float)
    dominance = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('Users', backref=db.backref('emotion_results', lazy=True))

class SlidingScaleResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    audio_file_name = db.Column(db.String(250), nullable=False)
    segment1_emotion = db.Column(db.String(100))
    segment2_emotion = db.Column(db.String(100))
    transcription = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('Users', backref=db.backref('sliding_scale_results', lazy=True))


 
def load_sentences(filepath="sentences.txt"):
    """
    Load sentences from a text file and return them as a list of dictionaries.
    Each line in the file should be formatted as:
    sentence1|sentence2|shifting_word
    """
    sentence_data = []
    try:
        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    sentence_data.append({
                        "sentence1": parts[0],
                        "sentence2": parts[1],
                        "shifting_word": parts[2]
                    })
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
    return sentence_data

def load_sentences_emotion(filepath="emotion-sentences.txt"):
    with open(filepath, "r") as file:
        return [line.strip() for line in file.readlines()]


def load_sentences_emphasis(filepath="emphasis-sentences.txt"):
    """
    Load sentences with emphasis words from a text file.
    Each line in the file should be formatted as:
    Sentence|Emphasis Word
    """
    sentences = []
    try:
        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) == 2:  # Ensure valid format
                    sentences.append({"sentence": parts[0], "emphasis_word": parts[1]})
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
    return sentences



# Initialize app with extension
db.init_app(app)
# Create database within app context
 
with app.app_context():
    db.create_all()

# Creates a user loader callback that returns the user object given an id
@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)

# Load model and processor
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

# Set device (use MPS for Apple Silicon or CPU fallback)
device = torch.device("cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

def convert_to_wav(input_path, output_path):
    """
    Converts audio file to .wav format.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise ValueError(f"Error converting audio to WAV format: {e}")

def calculate_pad_distance(predicted_pad, target_pad):
    return np.sqrt(sum((predicted_pad[i] - target_pad[i]) ** 2 for i in range(3)))

def calculate_score(predicted_pad, target_pad, predicted_emotion, target_emotion):
    """
    Calculate score for one segment based on exact emotion match and PAD closeness.
    """
    # Allocate 25 points for exact emotion match
    exact_match_score = 25 if predicted_emotion == target_emotion else 0

    # Calculate distance for PAD values and normalize to remaining 25 points
    max_distance = np.sqrt(3 * (6 ** 2))  # Max possible distance in 3D space (-3 to 3 for PAD)
    distance = np.sqrt(sum((predicted_pad[i] - target_pad[i]) ** 2 for i in range(3)))
    pad_closeness_score = max(0, 25 * (1 - (distance / max_distance)))

    # Total score
    return exact_match_score + pad_closeness_score


@app.route("/")
def home():
	# Render home.html on "/" route
	return render_template("main.html")

@app.route('/home')
def old_main_page():
    return render_template('home.html')

@app.route('/play')
@login_required
def play_mode():
    return render_template('play.html')

@app.route('/emotion-detection')
@login_required
def emotion_detection_page():
    results = EmotionResult.query.filter_by(user_id=current_user.id).order_by(EmotionResult.created_at.desc()).all()
    return render_template('emotion.html', results=results)
    #return render_template('emotion.html')  # Emotion detection page

@app.route("/classify-sliding-scale")
@login_required
def sliding_scale_page():
    # Load sentences from the file
    sentences = load_sentences("sentences.txt")
    if not sentences:
        return jsonify({"error": "No sentences available"}), 500

    # Select a random sentence and emotions
    selected_sentence = random.choice(sentences)
    emotions = list(emotion_pad_values.keys())
    emotion1, emotion2 = random.sample(emotions, 2)

    # Store selected sentence and emotions in session
    session["selected_sentence"] = selected_sentence
    session["emotion1"] = emotion1
    session["emotion2"] = emotion2

    # Query past results for the current user
    sliding_scale_results = SlidingScaleResult.query.filter_by(
        user_id=current_user.id
    ).order_by(SlidingScaleResult.created_at.desc()).all()



    return render_template(
        "sliding-scale.html",
        selected_sentence=selected_sentence,
        emotion1=emotion1,
        emotion2=emotion2,
        results=sliding_scale_results
    )

@app.route("/classify-sliding-scale-result", methods=["POST"])

@login_required

def sliding_scale_result():

    if "audio" not in request.files:

        return jsonify({"error": "No audio file uploaded"}), 400



    # Retrieve session variables

    selected_sentence = session.get("selected_sentence")

    random_emotion1 = session.get("emotion1")

    random_emotion2 = session.get("emotion2")

    if not selected_sentence:

        return jsonify({"error": "No selected sentence found in session"}), 500



    shifting_word = ' ' + selected_sentence["shifting_word"].lower()



    # Save uploaded audio

    audio_file = request.files["audio"]

    audio_filename = secure_filename(f"{current_user.id}_{audio_file.filename}")

    audio_path = os.path.join("static", "uploads", audio_filename)

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    audio_file.save(audio_path)



    try:

        # Process audio and retrieve emotions

        result = split_audio_on_word(audio_path, word=shifting_word)

        segment1_emotion = result["Segment 1 Emotion"]["Emotion"]

        segment2_emotion = result["Segment 2 Emotion"]["Emotion"]

        transcription = result["Transcription"]



        # Get PAD values for target and predicted emotions

        target_pad1 = emotion_pad_values[random_emotion1]

        target_pad2 = emotion_pad_values[random_emotion2]

        predicted_pad1 = [

            result["Segment 1 Emotion"]["Pleasure"],

            result["Segment 1 Emotion"]["Arousal"],

            result["Segment 1 Emotion"]["Dominance"],

        ]

        predicted_pad2 = [

            result["Segment 2 Emotion"]["Pleasure"],

            result["Segment 2 Emotion"]["Arousal"],

            result["Segment 2 Emotion"]["Dominance"],

        ]



        # Calculate scores

        segment1_score = calculate_score(predicted_pad1, target_pad1, segment1_emotion, random_emotion1)

        segment2_score = calculate_score(predicted_pad2, target_pad2, segment2_emotion, random_emotion2)

        overall_score = round(segment1_score + segment2_score, 2)



        # Save result to database

        sliding_scale_result = SlidingScaleResult(

            user_id=current_user.id,

            audio_file_name=audio_filename,

            segment1_emotion=segment1_emotion,

            segment2_emotion=segment2_emotion,

            transcription=transcription

        )

        db.session.add(sliding_scale_result)

        db.session.commit()



        # Return JSON response

        return jsonify({

            "Segment 1 Emotion": segment1_emotion,

            "Segment 2 Emotion": segment2_emotion,

            "Target Emotion 1": random_emotion1,

            "Target Emotion 2": random_emotion2,

            "Transcription": transcription,

            "Segment 1 Score": round(segment1_score, 2),

            "Segment 2 Score": round(segment2_score, 2),

            "Overall Score": overall_score

        })



    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
@login_required
def predict_emotion():
    # Check if the request contains an audio file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Save the uploaded audio file
    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the uploads directory exists

    print(request.files)

    audio_file = request.files['audio']
    audio_filename = secure_filename(f"{current_user.id}_{audio_file.filename}")
    audio_path = os.path.join(upload_dir, audio_filename)
    print(f"Audio filename: {audio_filename}")
    print(f"Audio path: {audio_path}")

    if audio_file:
        print(f"File size: {len(audio_file.read())} bytes")
        audio_file.seek(0)  # Reset the file pointer after checking size
    else:
        print("No file content")
    # Load and preprocess audio file

    try:
        audio_file.save(audio_path)  # Save the uploaded file
        print(f"File saved to {audio_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    try:
        audio_input, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {e}"}), 500

    # Prepare input for the model
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    try:
        with torch.no_grad():
            output = model(inputs['input_values'])
            arousal, dominance, valence = output[0].cpu().numpy()
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # Interpret PAD values and classify emotion
    pleasure = float(valence)  # Convert numpy.float32 to Python float
    arousal = float(arousal)
    dominance = float(dominance)
    emotion = classify_emotion([pleasure, arousal, dominance])

    # Save result to database
    result = EmotionResult(
        user_id=current_user.id,
        audio_file_name=audio_file.filename,
        emotion=str(emotion),
        pleasure = pleasure,
        arousal = arousal,
        dominance = dominance

    )
    db.session.add(result)
    db.session.commit()

    # Return the results as JSON
    return jsonify({
        "pleasure": round(pleasure, 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
        "emotion": emotion
    })

@app.route('/emphasis-detection')
@login_required
def emphasis_page():
    # Query past emphasis detection results for the user
    results = EmphasisResult.query.filter_by(user_id=current_user.id).order_by(EmphasisResult.created_at.desc()).all()
    return render_template('emphasis.html', results=results)

@app.route('/detect-emphasis', methods=['POST'])
@login_required
def detect_emphasis_api():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Save the uploaded audio file
    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the uploads directory exists

    print(request.files)

    audio_file = request.files['audio']
    audio_filename = secure_filename(f"{current_user.id}_{audio_file.filename}")
    audio_path = os.path.join(upload_dir, audio_filename)
    print(f"Audio filename: {audio_filename}")
    print(f"Audio path: {audio_path}")

    if audio_file:
        print(f"File size: {len(audio_file.read())} bytes")
        audio_file.seek(0)  # Reset the file pointer after checking size
    else:
        print("No file content")

    try:
        audio_file.save(audio_path)  # Save the uploaded file
        print(f"File saved to {audio_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        
    try:
        # Transcribe the audio and detect emphasized words
        num_words = int(request.form.get('numWords', 1))
        words_with_timestamps = transcribe_audio(audio_path)
        emphasized_words = detect_emphasis(audio_path, words_with_timestamps, num_words)
        # Save result to database
        result = EmphasisResult(
            user_id=current_user.id,
            audio_file_name=audio_file.filename,
            emphasized_words=",".join(emphasized_words),
            num_words=num_words
        )
        db.session.add(result)
        db.session.commit()

        #os.remove(audio_path)  # Clean up temporary file

        return jsonify({"emphasized_words": emphasized_words})
    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {e}"}), 500

@app.route('/register', methods=["GET", "POST"])
def register():
# If the user made a POST request, create a new user
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if the username is already taken
        existing_user = Users.query.filter_by(username=username).first()
        if existing_user:
            flash("Username is already taken, please choose a different one.", "danger")
            return redirect(url_for("register"))  # Redirect back to registration page

        # If the username is available, create the user
        user = Users(username=username, password=password)
        # Add the user to the database
        db.session.add(user)
        # Commit the changes made
        db.session.commit()
        # Once user account created, redirect them
        # to login route (created later on)
        return redirect(url_for("login"))
	# Renders sign_up template if user made a GET request
    return render_template("sign_up.html")

@app.route("/login", methods=["GET", "POST"])
def login():
	# If a post request was made, find the user by 
	# filtering for the username
    if request.method == "POST":
        user = Users.query.filter_by(
            username=request.form.get("username")).first()
		# Check if the password entered is the 
		# same as the user's password
        if user:
            if user.password == request.form.get("password"):
                # Use the login_user method to log in the user
                login_user(user)
                return redirect(url_for("home"))
            # Redirect the user back to the home
            else:
                flash("Invalid username or password", "danger")  # Show warning message
        else:
            flash("Username not found", "danger")  # No user found with that username
          
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/emotion-game")
@login_required
def emotion_game_page():
    # Load a random sentence-emotion pair from the file
    sentence_emotions = load_sentences_emotion("emotion-sentences.txt")
    if not sentence_emotions:
        return jsonify({"error": "No sentences available"}), 500

    # Randomly select a sentence and its corresponding target emotion
    selected_pair = random.choice(sentence_emotions)
    sentence, target_emotion = selected_pair.split("|")

    # Store the sentence and target emotion in the session for later use
    session["current_sentence"] = sentence
    session["target_emotion"] = target_emotion

    # Render the Emotion Game page
    return render_template("emotion-game.html", sentence=sentence, target_emotion=target_emotion)


@app.route("/emotion-game-result", methods=["POST"])
@login_required
def emotion_game_result():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Retrieve the target emotion from the session
    target_emotion = session.get("target_emotion")
    if not target_emotion:
        return jsonify({"error": "Target emotion not found in session"}), 500

    # Save the uploaded audio file
    audio_file = request.files["audio"]
    audio_filename = secure_filename(f"{current_user.id}_{audio_file.filename}")
    audio_path = os.path.join("static", "uploads", audio_filename)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    audio_file.save(audio_path)

    # Process the uploaded audio file and classify its emotion
    try:
        # Call the `predict_emotion` function
        result = predict_emotion_from_audio(audio_path)
        predicted_emotion = result["emotion"]
        predicted_pad = [result["pleasure"], result["arousal"], result["dominance"]]

        # Get PAD values for the target emotion
        target_pad = emotion_pad_values[target_emotion]

        # Calculate the score based on the predicted and target values
        score = calculate_score(predicted_pad, target_pad, predicted_emotion, target_emotion)
        
        session["emotion_score"] = round(score * 2, 2)

        # Return the result as JSON
        return jsonify({
            "sentence": session.get("current_sentence"),
            "target_emotion": target_emotion,
            "predicted_emotion": predicted_emotion,
            "score": round(score*2, 2),
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process the audio: {e}"}), 500


def predict_emotion_from_audio(audio_path):
    """
    Predict the emotion and PAD values from the given audio file.
    """
    # Load and preprocess audio file
    try:
        audio_input, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

    # Prepare input for the model
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Some models might not expect 'attention_mask'. Remove it if unnecessary.
    if "attention_mask" in inputs:
        inputs.pop("attention_mask")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    try:
        with torch.no_grad():
            outputs = model(inputs["input_values"])  # Only pass 'input_values'
            arousal, dominance, valence = outputs[0].cpu().numpy()
    except Exception as e:
        raise ValueError(f"Model inference failed: {e}")

    # Interpret PAD values and classify emotion
    pleasure = float(valence)
    arousal = float(arousal)
    dominance = float(dominance)
    emotion = classify_emotion([pleasure, arousal, dominance])

    return {
        "pleasure": pleasure,
        "arousal": arousal,
        "dominance": dominance,
        "emotion": emotion
    }

@app.route("/emphasis-game")
@login_required
def emphasis_game_page():
    # Load sentences with emphasis words
    sentences = load_sentences_emphasis("emphasis-sentences.txt")
    if not sentences:
        return jsonify({"error": "No sentences available"}), 500

    # Randomly select one sentence
    selected_sentence = random.choice(sentences)
    session["selected_sentence"] = selected_sentence

    # Pass only the selected sentence to the template
    return render_template(
        "emphasis-game.html", 
        sentence=selected_sentence["sentence"],
        emphasis_word=selected_sentence["emphasis_word"]
    )

@app.route("/emphasis-game-result", methods=["POST"])
@login_required
def emphasis_game_result():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Retrieve the selected sentence and emphasis word from the session
    selected_sentence = session.get("selected_sentence")
    if not selected_sentence:
        return jsonify({"error": "No selected sentence found in session"}), 500

    emphasis_word = selected_sentence["emphasis_word"].lower()

    # Save the uploaded audio file
    audio_file = request.files["audio"]
    audio_filename = secure_filename(f"{current_user.id}_{audio_file.filename}")
    audio_path = os.path.join("static", "uploads", audio_filename)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    audio_file.save(audio_path)

    # Detect emphasis in the audio file
    try:
        # Transcribe audio to get words with timestamps
        words_with_timestamps = transcribe_audio(audio_path)

        # Detect emphasis using the transcription
        emphasized_words = detect_emphasis(audio_path, words_with_timestamps, n=1)

        # Normalize case for comparison
        emphasized_words_lower = [word.lower() for word in emphasized_words]

        # Compare detected emphasis word with the target word
        if emphasis_word in emphasized_words_lower:
            score = 100
        else:
            score = 0

        session["emphasis_score"] = score
        
        # Return result as JSON
        return jsonify({
            "score": score,
            "sentence": selected_sentence["sentence"],
            "target_word": emphasis_word,
            "emphasized_words": emphasized_words,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process the audio: {e}"}), 500

@app.route("/sliding-scale-game")
@login_required
def sliding_scale_game():
    # Load a random sentence and emotions
    sentences = load_sentences("sentences.txt")
    if not sentences:
        return jsonify({"error": "No sentences available"}), 500

    selected_sentence = random.choice(sentences)
    emotions = list(emotion_pad_values.keys())
    emotion1, emotion2 = random.sample(emotions, 2)

    # Store the selected sentence and emotions in the session
    session["selected_sentence"] = selected_sentence
    session["emotion1"] = emotion1
    session["emotion2"] = emotion2

    return render_template(
        "sliding-scale-game.html",
        selected_sentence=selected_sentence,
        emotion1=emotion1,
        emotion2=emotion2,
    )

@app.route("/sliding-scale-game-result", methods=["POST"])
@login_required
def sliding_scale_game_result():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    selected_sentence = session.get("selected_sentence")
    emotion1 = session.get("emotion1")
    emotion2 = session.get("emotion2")

    if not selected_sentence:
        return jsonify({"error": "Session expired or invalid."}), 500

    shifting_word = ' ' + selected_sentence["shifting_word"].lower()
    audio_file = request.files["audio"]
    audio_filename = secure_filename(f"{current_user.id}_{audio_file.filename}")
    audio_path = os.path.join("static", "uploads", audio_filename)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    audio_file.save(audio_path)

    try:
        result = split_audio_on_word(audio_path, word=shifting_word)
        segment1_emotion = result["Segment 1 Emotion"]["Emotion"]
        segment2_emotion = result["Segment 2 Emotion"]["Emotion"]

        target_pad1 = emotion_pad_values[emotion1]
        target_pad2 = emotion_pad_values[emotion2]

        predicted_pad1 = [
            result["Segment 1 Emotion"]["Pleasure"],
            result["Segment 1 Emotion"]["Arousal"],
            result["Segment 1 Emotion"]["Dominance"],
        ]
        predicted_pad2 = [
            result["Segment 2 Emotion"]["Pleasure"],
            result["Segment 2 Emotion"]["Arousal"],
            result["Segment 2 Emotion"]["Dominance"],
        ]

        segment1_score = calculate_score(predicted_pad1, target_pad1, segment1_emotion, emotion1)
        segment2_score = calculate_score(predicted_pad2, target_pad2, segment2_emotion, emotion2)
        overall_score = round(segment1_score + segment2_score, 2)

        # Store score in session
        session["sliding_game_score"] = overall_score

        return jsonify({
            "Segment 1 Emotion": segment1_emotion,
            "Segment 2 Emotion": segment2_emotion,
            "Overall Score": overall_score,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/game-results")
@login_required
def game_results_page():
    # Get scores from the session
    emotion_score = session.get("emotion_score", 0)
    emphasis_score = session.get("emphasis_score", 0)
    sliding_game_score = session.get("sliding_game_score", 0)

    # Calculate the final weighted score
    final_score = (0.2 * emotion_score) + (0.2 * emphasis_score) + (0.6 * sliding_game_score)

    return render_template(
        "game-results.html",
        emotion_score=round(emotion_score, 2),
        emphasis_score=round(emphasis_score, 2),
        sliding_game_score=round(sliding_game_score, 2),
        final_score=round(final_score, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)
