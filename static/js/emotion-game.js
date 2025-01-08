// Select DOM elements
const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const classifyRecordingButton = document.getElementById("classifyRecording");
const recordedAudio = document.getElementById("recordedAudio");
const resultDiv = document.getElementById("result");
const resultContent = document.getElementById("resultContent");
const spinner = document.getElementById("spinner");

let mediaRecorder;
let audioChunks = [];

// Start Recording
recordButton.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioUrl = URL.createObjectURL(audioBlob);
            recordedAudio.src = audioUrl;
            recordedAudio.style.display = "block";
            classifyRecordingButton.style.display = "inline"; // Show classify button
        };

        mediaRecorder.start();
        recordButton.style.display = "none";
        stopButton.style.display = "inline";
        audioChunks = []; // Reset chunks for new recording
    } catch (error) {
        alert("Error accessing microphone: " + error.message);
    }
});

// Stop Recording
stopButton.addEventListener("click", () => {
    if (mediaRecorder) {
        mediaRecorder.stop();
        stopButton.style.display = "none";
        recordButton.style.display = "inline";
    }
});

// Classify Recording
classifyRecordingButton.addEventListener("click", async () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "recorded_audio.wav");

    spinner.style.display = "flex"; // Show loading spinner
    resultDiv.style.display = "none"; // Hide result while loading

    try {
        const response = await fetch("/emotion-game-result", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            resultContent.innerHTML = `
                <h3>Target Emotion: ${data.target_emotion}</h3>
                <h3>Predicted Emotion: ${data.predicted_emotion}</h3>
                <h3>Score: ${data.score} / 100</h3>
            `;
            resultDiv.style.display = "block"; // Show result
        } else {
            const error = await response.json();
            alert("Error: " + error.error);
        }
    } catch (err) {
        alert("Unexpected error: " + err.message);
    } finally {
        spinner.style.display = "none"; // Hide spinner
    }
});
