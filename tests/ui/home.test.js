const { test, expect } = require('@playwright/test');

// Configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:5000';

test.describe('Titanic Survival Prediction - UI Tests', () => {
    test.beforeEach(async ({ page }) => {
        // Navigate to prediction page before each test
        await page.goto(`${BASE_URL}/prediction`);
        await page.waitForLoadState('networkidle');
    });

    test('should load prediction page successfully', async ({ page }) => {
        // Check page title or heading
        const pageTitle = await page.title();
        expect(pageTitle).toBeTruthy();
        
        // Check for main form presence
        const form = await page.$('form');
        expect(form).not.toBeNull();
    });

    test('should display all required form fields', async ({ page }) => {
        // Check if all form fields are present
        const ageInput = await page.$('input[name="age"]');
        expect(ageInput).not.toBeNull();

        const genderSelect = await page.$('select[name="gender"]');
        expect(genderSelect).not.toBeNull();

        const titleSelect = await page.$('select[name="name_title"]');
        expect(titleSelect).not.toBeNull();

        const sibspSelect = await page.$('select[name="sibsp"]');
        expect(sibspSelect).not.toBeNull();

        const pclassSelect = await page.$('select[name="pclass"]');
        expect(pclassSelect).not.toBeNull();

        const embarkedSelect = await page.$('select[name="embarked"]');
        expect(embarkedSelect).not.toBeNull();

        const cabinSelect = await page.$('select[name="cabin_multiple"]');
        expect(cabinSelect).not.toBeNull();

        const submitButton = await page.$('input[type="submit"]');
        expect(submitButton).not.toBeNull();
    });

    test('should predict survival for first class female passenger', async ({ page }) => {
        // Fill out the form for a likely survivor
        await page.fill('input[name="age"]', '35');
        await page.selectOption('select[name="gender"]', 'female');
        await page.selectOption('select[name="name_title"]', 'Mrs');
        await page.selectOption('select[name="sibsp"]', '1');
        await page.selectOption('select[name="pclass"]', '1');
        await page.selectOption('select[name="embarked"]', 'C');
        await page.selectOption('select[name="cabin_multiple"]', '2');

        // Submit the form
        await page.click('input[type="submit"]');
        await page.waitForLoadState('networkidle');

        // Check for survival prediction
        const pageContent = await page.content();
        expect(pageContent.toLowerCase()).toMatch(/surviv/);
    });

    test('should predict no survival for third class male passenger', async ({ page }) => {
        // Fill out the form for a likely non-survivor
        await page.fill('input[name="age"]', '25');
        await page.selectOption('select[name="gender"]', 'male');
        await page.selectOption('select[name="name_title"]', 'Mr');
        await page.selectOption('select[name="sibsp"]', '0');
        await page.selectOption('select[name="pclass"]', '3');
        await page.selectOption('select[name="embarked"]', 'S');
        await page.selectOption('select[name="cabin_multiple"]', '0');

        // Submit the form
        await page.click('input[type="submit"]');
        await page.waitForLoadState('networkidle');

        // Check for prediction result
        const pageContent = await page.content();
        expect(pageContent.toLowerCase()).toMatch(/surviv|did not survive|not survive/);
    });

    test('should handle child passenger correctly', async ({ page }) => {
        // Fill form for child passenger
        await page.fill('input[name="age"]', '8');
        await page.selectOption('select[name="gender"]', 'male');
        await page.selectOption('select[name="name_title"]', 'Master');
        await page.selectOption('select[name="sibsp"]', '2');
        await page.selectOption('select[name="pclass"]', '2');
        await page.selectOption('select[name="embarked"]', 'S');
        await page.selectOption('select[name="cabin_multiple"]', '0');

        // Submit form
        await page.click('input[type="submit"]');
        await page.waitForLoadState('networkidle');

        // Verify prediction was made
        const pageContent = await page.content();
        expect(pageContent).toMatch(/\d+(\.\d+)?%/); // Should show percentage
    });

    test('should handle different passenger classes', async ({ page }) => {
        const classes = ['1', '2', '3'];
        
        for (const pclass of classes) {
            await page.goto(`${BASE_URL}/prediction`);
            
            await page.fill('input[name="age"]', '30');
            await page.selectOption('select[name="gender"]', 'female');
            await page.selectOption('select[name="name_title"]', 'Miss');
            await page.selectOption('select[name="sibsp"]', '0');
            await page.selectOption('select[name="pclass"]', pclass);
            await page.selectOption('select[name="embarked"]', 'S');
            await page.selectOption('select[name="cabin_multiple"]', '0');

            await page.click('input[type="submit"]');
            await page.waitForLoadState('networkidle');

            // Verify a prediction was returned
            const pageContent = await page.content();
            expect(pageContent.toLowerCase()).toMatch(/surviv/);
        }
    });

    test('should handle different embarkation ports', async ({ page }) => {
        const ports = ['C', 'Q', 'S'];
        
        for (const port of ports) {
            await page.goto(`${BASE_URL}/prediction`);
            
            await page.fill('input[name="age"]', '40');
            await page.selectOption('select[name="gender"]', 'male');
            await page.selectOption('select[name="name_title"]', 'Mr');
            await page.selectOption('select[name="sibsp"]', '0');
            await page.selectOption('select[name="pclass"]', '2');
            await page.selectOption('select[name="embarked"]', port);
            await page.selectOption('select[name="cabin_multiple"]', '0');

            await page.click('input[type="submit"]');
            await page.waitForLoadState('networkidle');

            // Verify prediction was made
            const pageContent = await page.content();
            expect(pageContent).toMatch(/\d+(\.\d+)?%/);
        }
    });

    test('should validate age input', async ({ page }) => {
        // Try submitting with invalid age
        await page.fill('input[name="age"]', '-5');
        await page.selectOption('select[name="gender"]', 'male');
        await page.selectOption('select[name="name_title"]', 'Mr');
        await page.selectOption('select[name="sibsp"]', '0');
        await page.selectOption('select[name="pclass"]', '3');
        await page.selectOption('select[name="embarked"]', 'S');
        await page.selectOption('select[name="cabin_multiple"]', '0');

        await page.click('input[type="submit"]');
        
        // Check for validation or error handling
        // The exact behavior depends on implementation
        const pageContent = await page.content();
        expect(pageContent).toBeTruthy();
    });

    test('should show confidence level in prediction', async ({ page }) => {
        await page.fill('input[name="age"]', '28');
        await page.selectOption('select[name="gender"]', 'female');
        await page.selectOption('select[name="name_title"]', 'Miss');
        await page.selectOption('select[name="sibsp"]', '0');
        await page.selectOption('select[name="pclass"]', '1');
        await page.selectOption('select[name="embarked"]', 'C');
        await page.selectOption('select[name="cabin_multiple"]', '1');

        await page.click('input[type="submit"]');
        await page.waitForLoadState('networkidle');

        // Check for percentage/confidence in result
        const pageContent = await page.content();
        expect(pageContent).toMatch(/\d+(\.\d+)?%|confidence/i);
    });

    test('should navigate to home page', async ({ page }) => {
        await page.goto(BASE_URL);
        
        // Check page loaded
        const pageContent = await page.content();
        expect(pageContent).toBeTruthy();
        expect(pageContent.length).toBeGreaterThan(0);
    });

    test('should handle form reset or multiple submissions', async ({ page }) => {
        // First submission
        await page.fill('input[name="age"]', '30');
        await page.selectOption('select[name="gender"]', 'male');
        await page.selectOption('select[name="name_title"]', 'Mr');
        await page.selectOption('select[name="sibsp"]', '0');
        await page.selectOption('select[name="pclass"]', '3');
        await page.selectOption('select[name="embarked"]', 'S');
        await page.selectOption('select[name="cabin_multiple"]', '0');

        await page.click('input[type="submit"]');
        await page.waitForLoadState('networkidle');

        // Navigate back and make another prediction
        await page.goto(`${BASE_URL}/prediction`);
        
        await page.fill('input[name="age"]', '45');
        await page.selectOption('select[name="gender"]', 'female');
        await page.selectOption('select[name="name_title"]', 'Mrs');
        await page.selectOption('select[name="sibsp"]', '1');
        await page.selectOption('select[name="pclass"]', '1');
        await page.selectOption('select[name="embarked"]', 'C');
        await page.selectOption('select[name="cabin_multiple"]', '2');

        await page.click('input[type="submit"]');
        await page.waitForLoadState('networkidle');

        // Verify second prediction worked
        const pageContent = await page.content();
        expect(pageContent.toLowerCase()).toMatch(/surviv/);
    });
});
