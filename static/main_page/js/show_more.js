
window.addEventListener('DOMContentLoaded', (event) => {
    const showMoreButton = document.getElementById('show-more-button');
    const hiddenImages = document.querySelectorAll('.col-lg-4.d-none');

    const imagesPerLoad = 6;
    let visibleCount = imagesPerLoad;

    if (hiddenImages.length > 0) {
        showMoreButton.style.display = 'block';
    }

    showMoreButton.addEventListener('click', () => {
        for (let i = visibleCount; i < visibleCount + imagesPerLoad; i++) {
            if (hiddenImages[i]) {
                hiddenImages[i].classList.remove('d-none');
            }
        }
        
        visibleCount += imagesPerLoad;

        if (visibleCount >= hiddenImages.length) {
            showMoreButton.style.display = 'none';
        }
    });
});