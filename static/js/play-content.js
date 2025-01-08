document.addEventListener("DOMContentLoaded", () => {
    const playSection = document.querySelector(".play-section");
    const contents = [
        "Welcome to the Play section! Here, you can engage in interactive challenges.",
        "First challenge: Practice modulating your emotions through voice.",
        "Second challenge: Try varying emphasis in a given script."
    ];
    let currentIndex = 0;

    const displayArea = document.createElement("p");
    displayArea.className = "play-text";
    playSection.appendChild(displayArea);

    const nextButton = document.createElement("button");
    nextButton.className = "next-button";
    nextButton.textContent = "Next";
    playSection.appendChild(nextButton);

    const inputField = document.createElement("input");
    inputField.className = "user-input";
    inputField.placeholder = "Enter your response here...";
    inputField.style.display = "none"; // Hide initially
    playSection.appendChild(inputField);

    const submitButton = document.createElement("button");
    submitButton.className = "submit-button";
    submitButton.textContent = "Submit";
    submitButton.style.display = "none"; // Hide initially
    playSection.appendChild(submitButton);

    const userResponses = []; // To store user responses

    function updateContent() {
        nextButton.disabled = true;
        // nextButton.style.display = "none";
        if (currentIndex < contents.length) {
            displayArea.textContent = contents[currentIndex];
            currentIndex++;

            // Show input field and submit button after the first challenge
            
            inputField.style.display = "inline-block";
            submitButton.style.display = "inline-block";
            
        } else {
            nextButton.style.display = "none";
            displayArea.textContent = "You've completed all the challenges. See your score below.";
            // nextButton.disabled = true;
            inputField.style.display = "none"; // Hide input field
            submitButton.style.display = "none"; // Hide submit button

            // Display all collected responses
            const responsesDisplay = document.createElement("div");
            responsesDisplay.className = "responses-display";
            responsesDisplay.innerHTML = `
                <h3>Your Responses:</h3>
                <ul>
                    ${userResponses.map(response => `<li>${response}</li>`).join('')}
                </ul>
            `;
            playSection.appendChild(responsesDisplay);
        }
    }
    nextButton.addEventListener("click", updateContent);

    submitButton.addEventListener("click", () => {
        const userInput = inputField.value.trim();
        if (userInput) {
            userResponses.push(userInput); // Save the response
            inputField.value = ""; // Clear the input field
            inputField.style.display = "none"; // Hide the input after submission
            submitButton.style.display = "none"; // Hide the submit button after submission
            nextButton.disabled = false; // Enable the next button
        } else {
            alert("Please enter a response before submitting.");
        }
    });
    
    updateContent(); // Initialize with the first content
});
