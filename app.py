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


@app.route("/")
def home():
	# Render home.html on "/" route
	return render_template("home.html")


@app.route('/index')
def main_page():
    return render_template('index.html')  # Main page with 3 buttons


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

    # Store the selected sentence and emotions in session
    session["selected_sentence"] = selected_sentence
    session["emotion1"] = emotion1
    session["emotion2"] = emotion2

    # Pass the selected sentence and emotions to the template
    return render_template(
        "sliding-scale.html",
        selected_sentence=selected_sentence,
        emotion1=emotion1,
        emotion2=emotion2
    )

@app.route("/classify-sliding-scale-result", methods=["POST"])
@login_required
def sliding_scale_result():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Retrieve the selected sentence and shifting word from session
    selected_sentence = session.get("selected_sentence")
    if not selected_sentence:
        return jsonify({"error": "No selected sentence found in session"}), 500

    shifting_word = selected_sentence["shifting_word"]
    shifting_word = ' ' + shifting_word.lower()

    # Save the uploaded audio file
    audio_file = request.files["audio"]
    audio_path = f"temp_{audio_file.filename}"
    audio_file.save(audio_path)

    try:
        # Pass the shifting word to split_audio_on_word
        result = split_audio_on_word(audio_path, word=shifting_word)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")
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


if __name__ == '__main__':
    app.run(debug=True)
