document.addEventListener('DOMContentLoaded', () => {
    console.log("Credit Card Approval UI scripts loaded.");

    // Handle Predict Form Submission & Animated Loader
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', (e) => {
            // Basic validation
            let isValid = true;
            const inputs = predictForm.querySelectorAll('input[required], select[required]');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    input.classList.add('is-invalid');
                    isValid = false;
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                e.preventDefault();
                alert("Please fill in all required fields.");
                return;
            }

            // Create loading overlay
            showLoader();
        });
    }
});

function showLoader() {
    const loaderHtml = `
        <div id="loading-overlay">
            <div class="loader-spinner"></div>
            <div class="loader-text" id="loader-status">Initializing Scoring Session...</div>
            <div class="loader-subtext">Executing ML Decision Tree & Random Forest classifiers</div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', loaderHtml);

    // Transition progress text to simulate server workload
    const statusText = document.getElementById('loader-status');
    const statuses = [
        "Analyzing Applicant Profile...",
        "Encoding Socio-Economic Categorical Features...",
        "Applying Standard Scaling Transformations...",
        "Running Random Forest & XGBoost Classifiers...",
        "Evaluating Credit Bureau Repayment Delinquencies...",
        "Calculating Approvals & Risk Classifications...",
        "Writing Session Logs to Database..."
    ];

    let step = 0;
    const interval = setInterval(() => {
        if (step < statuses.length && statusText) {
            statusText.innerText = statuses[step];
            step++;
        } else {
            clearInterval(interval);
        }
    }, 450);
}
