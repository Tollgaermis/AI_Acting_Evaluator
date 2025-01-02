document.addEventListener('DOMContentLoaded', () => {
    const toasts = document.querySelectorAll('.toast');
    toasts.forEach(toast => {
        // Listen for the end of the fade-out animation
        toast.addEventListener('animationend', (e) => {
            if (e.animationName === 'fadeOut') {
                toast.remove(); // Remove toast from DOM
            }
        });
    });
});