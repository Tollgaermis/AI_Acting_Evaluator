document.addEventListener('DOMContentLoaded', async () => {
    const resultsBody = document.getElementById('results-body');

    if (!resultsBody) {
        console.error('Results body not found!');
        return;
    }
  
    try {
      const response = await fetch('/api/game-results'); // Replace with your API endpoint
      const results = await response.json();
  
      results.forEach((result, index) => {
        const row = document.createElement('tr');

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
          <td>${result.created_at}</td>
        `;
        resultsBody.appendChild(row);
      });
    } catch (error) {
      console.error('Error fetching game results:', error);
    }
  });
  


