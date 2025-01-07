document.addEventListener("DOMContentLoaded", () => {
    const playSection = document.querySelector(".play-section");
    const contents = [
        "Welcome to the Play section! Here, you can engage in interactive challenges.",
        "First challenge: Practice modulating your emotions through voice.",
        "Second challenge: Try varying emphasis in a given script.",
        "Final challenge: Combine all skills for a dynamic performance!"
    ];
    let currentIndex = 0;

    const displayArea = document.createElement("p");
    displayArea.className = "play-text";
    playSection.appendChild(displayArea);

    const nextButton = document.createElement("button");
    nextButton.className = "next-button";
    nextButton.textContent = "Next";
    playSection.appendChild(nextButton);

    function updateContent() {
        if (currentIndex < contents.length) {
            displayArea.textContent = contents[currentIndex];
            currentIndex++;
        } else {
            displayArea.textContent = "You've completed all the challenges. Great job!";
            nextButton.disabled = true;
        }
    }

    nextButton.addEventListener("click", updateContent);
    updateContent(); // Initialize with the first content
});
