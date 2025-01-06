// File Upload: Handle audio file submission
document.getElementById('uploadButton').addEventListener('click', async (event) => {
    event.preventDefault();

    const audioFile = document.getElementById('audio').files[0];
    const numWords = document.getElementById('numWords') ? document.getElementById('numWords').value : 1;
    const resultDiv = document.getElementById('result');
    const spinner = document.getElementById('spinner');

    if (!audioFile) {
        alert("Please select an audio file.");
        return;
    }

    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('numWords', numWords);

    // Show loading spinner and hide result
    spinner.style.display = "flex";
    resultDiv.style.display = "";

    try {
        const response = await fetch('/detect-emphasis', {
            method: 'POST',
            body: formData,
        });

        // Redirect to login if necessary
        if (response.redirected) {
            window.location.href = response.url;
            return;
        }

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Emphasized Words:</h3>
                <ul>${data.emphasized_words.map(word => `<li>${word}</li>`).join('')}</ul>
            `;
        } else {
            const error = await response.json();
            resultDiv.innerHTML = `<p>Error: ${error.error}</p>`;
        }
    } catch (err) {
        resultDiv.innerHTML = `<p>Unexpected error: ${err.message}</p>`;
    } finally {
        spinner.style.display = "none";
    }
});

// Voice Recording Buttons
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const detectEmphasisRecordingButton = document.getElementById('detectEmphasisRecording');
const recordedAudio = document.getElementById('recordedAudio');
const resultDiv = document.getElementById('result');
const spinner = document.getElementById('spinner');

// Initialize variables for recording
let mediaRecorder;
let audioChunks = [];

// Start Recording
recordButton.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            recordedAudio.src = audioUrl;
            recordedAudio.style.display = 'block';
            detectEmphasisRecordingButton.style.display = 'inline';
        };

        mediaRecorder.start();
        recordButton.style.display = 'none';
        stopButton.style.display = 'inline';
        audioChunks = []; // Reset chunks for new recording
    } catch (error) {
        alert('Error accessing microphone: ' + error.message);
    }
});

// Stop Recording
stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    stopButton.style.display = 'none';
    recordButton.style.display = 'inline';
});

// Detect Emphasis in Recorded Audio
detectEmphasisRecordingButton.addEventListener('click', async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    const numWords = document.getElementById('numWords') ? document.getElementById('numWords').value : 1;
    formData.append('audio', audioBlob, 'recorded_audio.wav');
    formData.append('numWords', numWords);

    spinner.style.display = 'flex';
    resultDiv.style.display = 'none';

    try {
        const response = await fetch('/detect-emphasis', {
            method: 'POST',
            body: formData,
        });

        if (response.redirected) {
            window.location.href = response.url;
            return;
        }

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Emphasized Words:</h3>
                <ul>${data.emphasized_words.map(word => `<li>${word}</li>`).join('')}</ul>
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
        spinner.style.display = 'none';
    }
});
