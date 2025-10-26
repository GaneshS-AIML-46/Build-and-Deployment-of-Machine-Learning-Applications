document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loading = document.getElementById('loading');

    let selectedFile = null;

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // Handle file selection
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
            resultsSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Analyze button click
    analyzeBtn.addEventListener('click', function() {
        if (!selectedFile) {
            alert('Please select an image first.');
            return;
        }

        analyzeImage();
    });

    // Clear button click
    clearBtn.addEventListener('click', function() {
        selectedFile = null;
        fileInput.value = '';
        uploadArea.style.display = 'block';
        imagePreview.style.display = 'none';
        resultsSection.style.display = 'none';
        loading.style.display = 'none';
    });

    // Analyze image function
    function analyzeImage() {
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Show loading
        loading.style.display = 'block';
        resultsSection.style.display = 'none';

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Display results
            displayResults(data);
        })
        .catch(error => {
            loading.style.display = 'none';
            console.error('Error:', error);
            alert('An error occurred while analyzing the image.');
        });
    }

    // Display results function
    function displayResults(data) {
        const diagnosisValue = document.getElementById('diagnosisValue');
        const confidenceValue = document.getElementById('confidenceValue');
        const riskValue = document.getElementById('riskValue');
        const certaintyValue = document.getElementById('certaintyValue');
        const modelValue = document.getElementById('modelValue');

        diagnosisValue.textContent = data.result;
        confidenceValue.textContent = data.confidence;
        riskValue.textContent = data.risk_level;
        certaintyValue.textContent = data.certainty || 'Unknown';
        modelValue.textContent = data.model_used || 'Unknown';

        // Add appropriate classes for styling
        diagnosisValue.className = data.result.toLowerCase().includes('pneumonia') ? 'diagnosis-value pneumonia' : 'diagnosis-value normal';
        riskValue.className = data.risk_level.toLowerCase() === 'high' ? 'risk-value high' : 'risk-value low';

        resultsSection.style.display = 'block';
    }
});