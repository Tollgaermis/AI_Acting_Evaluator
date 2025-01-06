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
            console.log("Received data:", data);
    
            // Check if the data contains the expected details
            console.log("Segment 1 Details:", data["Segment 1 Emotion"]?.Details);
            console.log("Segment 2 Details:", data["Segment 2 Emotion"]?.Details);
            resultDiv.innerHTML = `
                <h3>Emotion Analysis Results</h3>
                <p><strong>Segment 1 (${data["Segment 1 Emotion"].Emotion}):</strong> ${data["Segment 1 Emotion"].Details}</p>
                <p><strong>Segment 2 (${data["Segment 2 Emotion"].Emotion}):</strong> ${data["Segment 2 Emotion"].Details}</p>
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
