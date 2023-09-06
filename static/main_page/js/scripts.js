/*!
* Start Bootstrap - Agency v7.0.12 (https://startbootstrap.com/theme/agency)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-agency/blob/master/LICENSE)
*/
//
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    //  Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            rootMargin: '0px 0px -40%',
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

});


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

const listGroups = document.querySelectorAll('.list-group');
listGroups.forEach(listGroup => {
  const checkboxes = listGroup.querySelectorAll('.form-check-input');
  checkboxes.forEach(checkbox => {
    checkbox.addEventListener('click', () => {
      if (checkbox.checked) {
        checkboxes.forEach(cb => {
          if (cb !== checkbox) {
            cb.checked = false;
          }
        });
      }
    });
  });
});

function sort_img() {
    var form = document.getElementById("contactForm");
    var inputElement = document.getElementById("inputField");

    if (inputElement.value == 1) {
        inputElement.setAttribute("value", "-1");
    } else {
        inputElement.setAttribute("value", "1");
    }

    form.submit();
}
function update() {
    var form = document.getElementById("contactForm");
    var inputElement = document.getElementById("inputField");

    inputElement.setAttribute("value", "2");
    
    form.submit();
}
function update_tr() {
    var form = document.getElementById("contactForm");
    var inputElement = document.getElementById("inputField");

    inputElement.setAttribute("value", "3");
    
    form.submit();
}

function autoScroll() {
    const dom = document.getElementById("portfolio")
    console.log(dom.offsetTop);
    setTimeout(function() {
        window.scrollTo({
            top: dom.offsetTop,
            behavior: 'smooth',
        });
    }, 50);
}