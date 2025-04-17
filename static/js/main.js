// Main JavaScript for all pages

// Update normal ranges based on gender selection
document.addEventListener('DOMContentLoaded', function() {
    // Only run on the report entry page
    if (document.getElementById('blood-report-form')) {
        const genderRadios = document.querySelectorAll('input[name="gender"]');
        
        genderRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                updateNormalRanges(this.value);
            });
        });
        
        // Initialize with male ranges
        updateNormalRanges('male');
        
        // Handle form submission
        document.getElementById('blood-report-form').addEventListener('submit', function(e) {
            e.preventDefault();
            submitBloodReport();
        });
    }
    
    // Load report data on results page
    if (document.getElementById('results-table-body')) {
        loadReportResults();
    }
});

function updateNormalRanges(gender) {
    const rangeCells = document.querySelectorAll('.normal-range[data-gender]');
    
    rangeCells.forEach(cell => {
        if (gender === 'male' && cell.dataset.gender === 'male') {
            cell.textContent = cell.dataset.maleRange || '13.5 - 17.5';
        } else if (gender === 'female' && cell.dataset.gender === 'female') {
            cell.textContent = cell.dataset.femaleRange || '12.0 - 15.5';
        }
    });
}


// Blood report submission and predicting the results from the ML model using Flask API
function submitBloodReport() {
    const age = document.getElementById('age').value;
    const gender = document.querySelector('input[name="gender"]:checked').value;
    const testResults = {};
    
    // Validate age
    if (!age || isNaN(age) || age < 1 || age > 120) {
        showError('Please enter a valid age between 1 and 120');
        return;
    }
    
    // Collect test results
    const inputs = document.querySelectorAll('#blood-report-form input[type="number"]');
    let hasResults = false;
    
    inputs.forEach(input => {
        if (input.value) {
            testResults[input.name] = parseFloat(input.value);
            hasResults = true;
        }
    });
    
    if (!hasResults) {
        showError('Please enter at least one test result');
        return;
    }
    
    // Send data to Flask backend
    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            gender,
            age: parseInt(age),
            testResults
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Analysis failed'); 
        }
        return response.json();
    })
    .then(data => {
        // Store results in sessionStorage and redirect
        sessionStorage.setItem('bloodReportData', JSON.stringify(data));
        window.location.href = '/results';
    })
    .catch(error => {
        showError('Failed to analyze report. Please try again.'); // if any error occurs in the backend this message will be shown at the top of enter report page
        console.error('Error:', error);
    });
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // Scroll to error message
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

// Results page functions
function loadReportResults() {
    const reportData = JSON.parse(sessionStorage.getItem('bloodReportData'));
    
    if (!reportData) {
        window.location.href = '/enter-report';
        return;
    }
    
    // Display patient details
    document.getElementById('patient-gender').textContent = reportData.gender === 'male' ? 'Male' : 'Female';
    document.getElementById('patient-age').textContent = reportData.age;
    
    // Display results
    displayResults(reportData.analysis);
    displayPredictions(reportData.ml_predictions);
    displayRecommendations(reportData.recommendations);
}

function displayResults(analysis) {
    const tableBody = document.getElementById('results-table-body');
    tableBody.innerHTML = '';
    
    let hasAbnormalities = false;
    
    
    for (const testName in analysis) {
        const result = analysis[testName];
        
        const row = document.createElement('tr');
        
        // Test Name
        const nameCell = document.createElement('td');
        nameCell.textContent = testName;
        row.appendChild(nameCell);
        
        // Value
        const valueCell = document.createElement('td');
        valueCell.textContent = result.value;
        row.appendChild(valueCell);
        
        // Normal Range
        const rangeCell = document.createElement('td');
        rangeCell.textContent = result.reference_range;
        row.appendChild(rangeCell);
        
        // Status
        const statusCell = document.createElement('td');
        const statusSpan = document.createElement('span');
        statusSpan.textContent = result.status.toUpperCase();
        statusSpan.className = result.status === 'normal' ? 'status-normal' : 'status-abnormal';
        
        if (result.status !== 'normal') {
            hasAbnormalities = true;
        }
        
        statusCell.appendChild(statusSpan);
        row.appendChild(statusCell);
        
        tableBody.appendChild(row);
    }
    
    // Show abnormalities section if any
    // if (hasAbnormalities) {
    //     document.getElementById('abnormalities-section').style.display = 'block';
    //     displayAbnormalities(analysis);
    // }
}

function displayAbnormalities(analysis) {
    const abnormalitiesList = document.getElementById('abnormalities-list');
    abnormalitiesList.innerHTML = '';
    
    for (const [testName, result] of Object.entries(analysis)) {
        if (result.status !== 'normal') {
            const li = document.createElement('li');
            li.innerHTML = `
                <strong>${testName}:</strong> ${result.value} (${result.status})
                <br>
                <small>Normal range: ${result.reference_range}</small>
            `;
            abnormalitiesList.appendChild(li);
        }
    }
}

// Display ML predictions along with accuracy percentage
function displayPredictions(ml_predictions) {
    const predictionsList = document.getElementById('predictions-list');
    predictionsList.innerHTML = '';

    for (const [testName, prediction] of Object.entries(ml_predictions)) {
        const li = document.createElement('li');
        

        li.innerHTML = `<strong>${testName}:</strong> ${prediction.toFixed(2)}%`;
        li.className = testName === 'Normal' ? 'status-normal predictions-list-normal' : 'status-abnormal predictions-list-abnormal';
        // li.style.color = testName === 'Normal' ? '#28a745' : '#dc3545';
        predictionsList.appendChild(li);
    }
    
    document.getElementById('predictions-section').style.display = 'block';


}


// Display Rule based recommendations based on analysis results
function displayRecommendations(recommendations) {
    const recommendationsContent = document.getElementById('recommendations-content');
    recommendationsContent.innerHTML = '';
    
    recommendations.forEach(rec => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        
        const h3 = document.createElement('h3');
        h3.textContent = rec.title;
        div.appendChild(h3);
        
        if (rec.description) {
            const p = document.createElement('p');
            p.textContent = rec.description;
            div.appendChild(p);
        }
        
        const ul = document.createElement('ul');
        rec.items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            ul.appendChild(li);
        });
        
        div.appendChild(ul);
        recommendationsContent.appendChild(div);
    });
}

// PDF Download
if (document.getElementById('download-pdf')) {
    document.getElementById('download-pdf').addEventListener('click', function() {
        generatePDF();
    });
}

function generatePDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    const reportData = JSON.parse(sessionStorage.getItem('bloodReportData'));
    
    // Add title
    doc.setFontSize(20);
    doc.text('Blood Report Analysis', 105, 20, { align: 'center' });
    
    // Add date
    doc.setFontSize(12);
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 105, 30, { align: 'center' });
    
    // Patient details
    doc.setFontSize(14);
    doc.text('Patient Details:', 20, 45);
    doc.setFontSize(12);
    doc.text(`Gender: ${reportData.gender === 'male' ? 'Male' : 'Female'}`, 20, 55);
    doc.text(`Age: ${reportData.age}`, 20, 65);
    
    // Test results table
    doc.setFontSize(14);
    doc.text('Test Results:', 20, 80);
    
    let y = 90;
    doc.setFontSize(12);
    doc.text('Test', 20, y);
    doc.text('Value', 70, y);
    doc.text('Normal Range', 100, y);
    doc.text('Status', 150, y);
    y += 10;
    
    for (const [testName, result] of Object.entries(reportData.analysis)) {
        doc.text(testName, 20, y);
        doc.text(result.value.toString(), 70, y);
        doc.text(result.reference_range, 100, y);
        doc.text(result.status.toUpperCase(), 150, y);
        y += 10;
        
        if (y > 250) {
            doc.addPage();
            y = 20;
        }
    }
    
    // Predictions
    doc.setFontSize(14);
    doc.text('ML Predictions:', 20, y + 10);
    y += 20;
    doc.setFontSize(12);
    
    for (const [testName, prediction] of Object.entries(reportData.ml_predictions)) {
        doc.setFont('helvetica', 'bold');
        doc.text(testName + ':', 20, y);
        doc.setFont('helvetica', 'normal');
        doc.text(`${prediction.toFixed(2)}%`, 90, y);
        y += 10;
    }

    // Recommendations
    doc.setFontSize(14);
    doc.text('Recommendations:', 20, y + 10);
    y += 20;
    doc.setFontSize(12);
    
    reportData.recommendations.forEach(rec => {
        doc.setFont('helvetica', 'bold');
        doc.text(rec.title, 20, y);
        doc.setFont('helvetica', 'normal');
        y += 10;
        
        if (rec.description) {
            doc.text(rec.description, 25, y);
            y += 10;
        }
        
        rec.items.forEach(item => {
            doc.text(`• ${item}`, 25, y);
            y += 10;
            
            if (y > 250) {
                doc.addPage();
                y = 20;
            }
        });
        
        y += 5;
    });


    
    // Footer note
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text('Note: This report is generated automatically and should not replace professional medical advice.', 105, 280, { align: 'center' });
    doc.text('Consult your healthcare provider for proper diagnosis and treatment.', 105, 285, { align: 'center' });
    
    // Save the PDF
    doc.save('blood_report_analysis.pdf');
}
