function removePlaceholderOption(selectElement) {
    var placeholderOption = selectElement.querySelector('.placeholder-option');
    if (placeholderOption) {
        placeholderOption.remove();
    }
}

// Show loading overlay on form submit
document.addEventListener('DOMContentLoaded', function () {
    var form = document.querySelector('form');
    var overlay = document.getElementById('loading-overlay');
    if (form && overlay) {
        form.addEventListener('submit', function () {
            overlay.classList.add('active');
            overlay.setAttribute('aria-hidden', 'false');
        });
    }
});

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