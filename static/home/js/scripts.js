/*!
* Start Bootstrap - Coming Soon v6.0.7 (https://startbootstrap.com/theme/coming-soon)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-coming-soon/blob/master/LICENSE)
*/
// This file is intentionally blank
// Use this file to add JavaScript to your project

document.querySelectorAll(".drop-zone__input").forEach((inputElement) => {
	const dropZoneElement = inputElement.closest(".drop-zone");
    const promptText = dropZoneElement.querySelector(".drop-zone__prompt").textContent;

	dropZoneElement.addEventListener("click", (e) => {
		inputElement.click();
	});

	inputElement.addEventListener("change", (e) => {
		if (inputElement.files.length) {
			updateThumbnail(dropZoneElement, inputElement.files[0]);
		}
	});

	dropZoneElement.addEventListener("dragover", (e) => {
		e.preventDefault();
		dropZoneElement.classList.add("drop-zone--over");
	});

	["dragleave", "dragend"].forEach((type) => {
		dropZoneElement.addEventListener(type, (e) => {
			dropZoneElement.classList.remove("drop-zone--over");
		});
	});

	dropZoneElement.addEventListener("drop", (e) => {
		e.preventDefault();

		if (e.dataTransfer.files.length) {
			inputElement.files = e.dataTransfer.files;
			updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
		}

		dropZoneElement.classList.remove("drop-zone--over");
	});
});

/**
 * Updates the thumbnail on a drop zone element.
 *
 * @param {HTMLElement} dropZoneElement
 * @param {File} file
 */
function updateThumbnail(dropZoneElement, file) {
    // Remove the prompt and show loading spinner
    if (dropZoneElement.querySelector(".drop-zone__prompt")) {
        dropZoneElement.querySelector(".drop-zone__prompt").remove();

        const loadingText = document.createElement("span");
        loadingText.classList.add("drop-zone__prompt");
        loadingText.style.marginLeft = "10px";
        loadingText.innerHTML = file.name;

        dropZoneElement.appendChild(loadingText);

		var btn = document.getElementById("submit-button");
		btn.disabled = false;
    }
}

function loading(dropZoneElement) {
    // Remove the prompt and show loading spinner
    if (dropZoneElement.querySelector(".drop-zone__prompt")) {
        dropZoneElement.querySelector(".drop-zone__prompt").remove();

		const loadingSpinner = document.createElement("span");
        loadingSpinner.classList.add("spinner-border", "spinner-border");
        loadingSpinner.setAttribute("role", "status");
        loadingSpinner.setAttribute("aria-hidden", "true");
        loadingSpinner.style.animationDuration = "2s";

        dropZoneElement.appendChild(loadingSpinner);

		const loadingText = document.createElement("span");
        loadingText.classList.add("drop-zone__prompt");
        loadingText.style.marginLeft = "10px";
        loadingText.innerHTML = "Loading...";

        dropZoneElement.appendChild(loadingText);

		var btn = document.getElementById("submit-button");
		btn.disabled = true;
		var form = document.getElementById("contactForm");
		form.submit();
    }
}

function autoSubmit() {
	var form = document.getElementById("contactForm");
	form.submit();
}