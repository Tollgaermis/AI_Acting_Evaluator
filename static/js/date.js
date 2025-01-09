document.addEventListener("DOMContentLoaded", () => {
    // Function to format the date in "dd.MM.yyyy HH.mm" format
    function formatDate(dateString) {
        const date = new Date(dateString);

        // Extract date components
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0'); // Months are 0-based
        const year = date.getFullYear();

        // Extract time components
        const hours = String(date.getHours()).padStart(2, '0'); // 24-hour format
        const minutes = String(date.getMinutes()).padStart(2, '0');

        // Combine into desired format
        return `${day}.${month}.${year} ${hours}.${minutes}`;
    }

    // Select all elements with the "date-cell" class
    const dateCells = document.querySelectorAll(".date-cell");

    // Iterate through each date cell and update its content
    dateCells.forEach((cell) => {
        const originalDate = cell.textContent.trim(); // Get the original date string
        const formattedDate = formatDate(originalDate); // Format the date
        cell.textContent = formattedDate; // Update the cell content
    });
});
