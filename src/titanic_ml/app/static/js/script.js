function removePlaceholderOption(selectElement) {
    // Get the placeholder option
    var placeholderOption = selectElement.querySelector('.placeholder-option');

    // Remove the placeholder option from the dropdown list
    if (placeholderOption) {
        placeholderOption.remove();
    }
}

// Prefill form with movie character data
function prefillCharacter(character) {
    var characters = {
        jack: { age: 20, gender: 'male', name_title: 'Mr', sibsp: '0', pclass: '3', embarked: 'S', cabin_multiple: '0' },
        rose: { age: 17, gender: 'female', name_title: 'Miss', sibsp: '0', pclass: '1', embarked: 'S', cabin_multiple: '1' }
    };

    var data = characters[character];
    if (!data) return;

    document.querySelector('input[name="age"]').value = data.age;
    document.querySelector('select[name="gender"]').value = data.gender;
    document.querySelector('select[name="name_title"]').value = data.name_title;
    document.querySelector('select[name="sibsp"]').value = data.sibsp;
    document.querySelector('select[name="pclass"]').value = data.pclass;
    document.querySelector('select[name="embarked"]').value = data.embarked;
    document.querySelector('select[name="cabin_multiple"]').value = data.cabin_multiple;
}

// Smooth scroll to anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Toggle modal visibility
const modalBtn = document.querySelector('.modal-btn');
const modal = document.querySelector('.modal');

modalBtn.addEventListener('click', () => {
    modal.classList.toggle('show-modal');
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.classList.remove('show-modal');
    }
});

// Close modal when pressing Escape key
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('show-modal')) {
        modal.classList.remove('show-modal');
    }
});