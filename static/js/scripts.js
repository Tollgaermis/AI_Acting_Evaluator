document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData();
    const audioFile = document.getElementById('audio').files[0];
    const resultDiv = document.getElementById('result');
    const spinner = document.getElementById('spinner');

    if (!audioFile) {
        alert("Please select an audio file.");
        return;
    }

    formData.append('audio', audioFile);

    // Show loading spinner and hide result
    spinner.style.display = "flex"; // Display spinner overlay
    resultDiv.style.display = ""; // Hide result block while loading

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

         if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Predicted Emotion: <span>${data.emotion}</span></h3>
                <p><strong>Pleasure:</strong> ${data.pleasure}</p>
                <p><strong>Arousal:</strong> ${data.arousal}</p>
                <p><strong>Dominance:</strong> ${data.dominance}</p>
            `;
        } else {
            const error = await response.json();
            resultDiv.innerHTML = `<p>Error: ${error.error}</p>`;
        }
    } catch (err) {
        resultDiv.innerHTML = `<p>Unexpected error: ${err.message}</p>`;
    } finally {
        // Hide loading spinner
        spinner.style.display = "none";
    }
});


// Get references to UI elements
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const classifyRecordingButton = document.getElementById('classifyRecording');
const recordedAudio = document.getElementById('recordedAudio');
const resultDiv = document.getElementById('result');
const spinner = document.getElementById('spinner');

// Initialize variables for recording
let mediaRecorder;
let audioChunks = [];

// Start recording
recordButton.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            console.log("Data available:", event.data);
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            console.log("Recording stopped. Processing audio...");
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            recordedAudio.src = audioUrl;
            recordedAudio.style.display = 'block';
            classifyRecordingButton.style.display = 'block';
            console.log("Audio Blob:", audioBlob);
        };

        mediaRecorder.start();
        console.log("Recording started...");
        recordButton.style.display = 'none';
        stopButton.style.display = 'inline';
        audioChunks = []; // Reset chunks for new recording
    } catch (error) {
        alert('Error accessing microphone: ' + error.message);
    }
});

// Stop recording
stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    stopButton.style.display = 'none';
    recordButton.style.display = 'inline';
});

// Classify the recorded audio
classifyRecordingButton.addEventListener('click', async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recorded_audio.wav');

    spinner.style.display = 'flex'; // Show spinner
    resultDiv.style.display = 'none'; // Hide previous results

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Predicted Emotion: <span>${data.emotion}</span></h3>
                <p><strong>Pleasure:</strong> ${data.pleasure}</p>
                <p><strong>Arousal:</strong> ${data.arousal}</p>
                <p><strong>Dominance:</strong> ${data.dominance}</p>
            `;
            resultDiv.style.display = 'block';
        } else {
            const error = await response.json();
            resultDiv.innerHTML = `<p>Error: ${error.error}</p>`;
            resultDiv.style.display = 'block';
        }
    } catch (err) {
        resultDiv.innerHTML = `<p>Unexpected error: ${err.message}</p>`;
        resultDiv.style.display = 'block';
    } finally {
        spinner.style.display = 'none'; // Hide spinner
    }
});