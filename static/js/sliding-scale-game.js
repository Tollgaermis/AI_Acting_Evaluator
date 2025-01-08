const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const nextButton = document.getElementById("nextButton");
const classifyRecordingButton = document.getElementById("classifyRecording");
const recordedAudio = document.getElementById("recordedAudio");
const resultDiv = document.getElementById("result");
const spinner = document.getElementById("spinner");

let mediaRecorder;
let audioChunks = [];

nextButton.disabled = true;

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
        audioChunks = [];
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
        const response = await fetch("/sliding-scale-game-result", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Segment 1: ${data["Target Emotion 1"]} → ${data["Segment 1 Emotion"]}</h3>
                <h3>Segment 2: ${data["Target Emotion 2"]} → ${data["Segment 2 Emotion"]}</h3>
                <p><strong>Transcription:</strong> ${data["Transcription"]}</p>
            `;
            nextButton.disabled = false;
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


nextButton.addEventListener('click', async () => {
    try {
        // Trigger the POST request to register the game results
        const response = await fetch("/register-game-results", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
        });

        if (response.ok) {
            // If the request was successful, redirect to the game results page
            window.location.href = "/game-results";  // Adjust URL as needed
        } else {
            // Handle the error if the request failed
            const error = await response.json();
            console.error('Error registering game results:', error);
            alert('Error registering game results');
        }
    } catch (error) {
        console.error('Unexpected error:', error);
        alert('Unexpected error occurred');
    }
});
