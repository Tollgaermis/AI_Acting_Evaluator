const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const classifyRecordingButton = document.getElementById("classifyRecording");
const recordedAudio = document.getElementById("recordedAudio");
const resultDiv = document.getElementById("result");
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
            classifyRecordingButton.style.display = "inline"; // Ensure classify button appears
        };

        mediaRecorder.start();
        recordButton.style.display = "none";
        stopButton.style.display = "inline";
        audioChunks = []; // Reset audio chunks for new recording
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

    spinner.style.display = "flex";
    resultDiv.style.display = "none";

    try {
        const response = await fetch("/classify-sliding-scale-result", {
            method: "POST",
            body: formData,
        });

        if (response.redirected) {
            window.location.href = response.url;
            return;
        }

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Emotion Analysis Results</h3>
                <p>
                    <strong>Segment 1 (${data["Target Emotion 1"]}):</strong>
                    ${data["Segment 1 Emotion"]} 
                    (${data["Segment 1 Score"]} points)
                </p>
                <p>
                    <strong>Segment 2 (${data["Target Emotion 2"]}):</strong>
                    ${data["Segment 2 Emotion"]}
                    (${data["Segment 2 Score"]} points)
                </p>
                <p><strong>Overall Score:</strong> ${data["Overall Score"]} points</p>
                <p><strong>Full Transcription:</strong> ${data["Transcription"]}</p>
            `;
        } else {
            const error = await response.json();
            resultDiv.innerHTML = `<p>Error: ${error.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p>Unexpected error: ${error.message}</p>`;
    } finally {
        spinner.style.display = "none";
        resultDiv.style.display = "block";
    }
});
