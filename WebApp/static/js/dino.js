document.addEventListener("DOMContentLoaded", function() {
    const loadingOverlay = document.getElementById('loading-overlay');

    setTimeout(function() {
        loadingOverlay.style.display = 'none';
    }, 6000);
});
