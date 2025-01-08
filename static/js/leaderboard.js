document.addEventListener('DOMContentLoaded', async () => {
    const resultsBody = document.getElementById('results-body');

    if (!resultsBody) {
        console.error('Results body not found!');
        return;
    }
  
    try {
      const response = await fetch('/api/game-results'); // Replace with your API endpoint
      const results = await response.json();
  
      results.forEach(result => {
        const row = document.createElement('tr');
        row.innerHTML = `
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
  