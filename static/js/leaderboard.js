document.addEventListener('DOMContentLoaded', async () => {
    const resultsBody = document.getElementById('results-body');

    if (!resultsBody) {
        console.error('Results body not found!');
        return;
    }
    
    function formatDate(dateString) {
      // Parse the date string
      const date = new Date(dateString);
  
      if (isNaN(date)) {
          console.error("Invalid date:", dateString); // Log invalid dates
          return dateString; // Return the original string if invalid
      }
  
      // Adjust for Turkey's time zone (UTC+3)
      const offsetDate = new Date(date.getTime() + 6 * 60 * 60 * 1000); 
  
      // Extract date components
      const day = String(offsetDate.getUTCDate()).padStart(2, '0');
      const month = String(offsetDate.getUTCMonth() + 1).padStart(2, '0'); // Months are 0-based
      const year = offsetDate.getUTCFullYear();
  
      // Extract time components
      const hours = String(offsetDate.getUTCHours()).padStart(2, '0'); // Adjusted for UTC+3
      const minutes = String(offsetDate.getUTCMinutes()).padStart(2, '0');
  
      // Combine into desired format
      return `${day}.${month}.${year} ${hours}.${minutes}`;
      }
  
  
  
  
    try {
      const response = await fetch('/api/game-results'); // Replace with your API endpoint
      const results = await response.json();
  
      results.forEach((result, index) => {
        const row = document.createElement('tr');
        const formattedDate = formatDate(result.created_at);

        // Add row index as the first cell
        // const indexCell = document.createElement('td');
        // indexCell.textContent = index + 1;  // Adding 1 to start index from 1
        // row.appendChild(indexCell);

        row.innerHTML = `
          <td>${index+1}</td>
          <td>${result.username}</td>
          <td>${result.emotion_score}</td>
          <td>${result.emphasis_score}</td>
          <td>${result.sliding_game_score}</td>
          <td>${result.overall_score}</td>
          <td>${formattedDate}</td>        `;
        resultsBody.appendChild(row);
      });
    } catch (error) {
      console.error('Error fetching game results:', error);
    }
  });
  


