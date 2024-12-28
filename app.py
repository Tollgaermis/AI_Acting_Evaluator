from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
import numpy as np
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
def emotion_detection_page():
    return render_template('emotion.html')  # Emotion detection page

@app.route("/classify-sliding-scale")
def sliding_scale_page():
    return render_template("sliding-scale.html")  # Sliding scale page


@app.route("/classify-sliding-scale-result", methods=["POST"])
@login_required
def sliding_scale_result():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Save uploaded audio
    audio_file = request.files["audio"]
    audio_path = f"temp_{audio_file.filename}"
    audio_file.save(audio_path)

    try:
        # Use " But" explicitly
        result = split_audio_on_word(audio_path, word=" But")
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
    audio_file = request.files['audio']
    audio_path = f"temp_{audio_file.filename}"
    audio_file.save(audio_path)

    # Load and preprocess audio file
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

    # Return the results as JSON
    return jsonify({
        "pleasure": round(pleasure, 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
        "emotion": emotion
    })


@app.route('/emphasis-detection')
def emphasis_page():
    return render_template('emphasis.html')

@app.route('/detect-emphasis', methods=['POST'])
@login_required
def detect_emphasis_api():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Save the uploaded audio file
    audio_file = request.files['audio']
    n_words = int(request.form.get('numWords', 1))
    audio_path = secure_filename(f"temp_{audio_file.filename}")
    audio_file.save(audio_path)

    try:
        # Transcribe the audio and detect emphasized words
        words_with_timestamps = transcribe_audio(audio_path)
        emphasized_words = detect_emphasis(audio_path, words_with_timestamps, n_words)
        os.remove(audio_path)  # Clean up temporary file

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
