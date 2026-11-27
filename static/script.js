const extractButton = document.getElementById('extract-button');
const text = document.getElementById('text');
const pipelineStage = document.getElementById('pipeline-stage');
const clearButton = document.getElementById('clear-button');
const fileInput = document.getElementById('file-input');
const sendApiButton = document.getElementById('send-api-button');
const pipelineStatus = document.getElementById('pipeline-status') || document.getElementById('api-status');
const reviewStage = document.getElementById('review-stage');
const resultsBody = document.getElementById('results-body');
const editStage = document.getElementById('edit-stage');
const outputStage = document.getElementById('output-stage');
const editText = document.getElementById('edit-text');
const outputText = document.getElementById('output-text');
const correctionYes = document.getElementById('correction-yes');
const correctionNo = document.getElementById('correction-no');
const saveEditButton = document.getElementById('save-edit-button');
const labelMode = document.getElementById('label-mode');
const blackMode = document.getElementById('black-mode');
const copyButton = document.getElementById('copy-button');
const downloadButton = document.getElementById('download-button');
const outputStatus = document.getElementById('output-status');
const inputStatus = document.getElementById('input-status');

let entities = [];
let originalText = '';
let fileProcessing = false;

function hasReadableText(text) {
    if (!text || text.trim().length < 20) {
        return false;
    }

    if (text.includes('\uFFFD')) {
        return false;
    }

    const printableCharacters = text.match(/[A-Za-z0-9\s.,!?'"():;\-]/g);
    const printableRatio = printableCharacters
        ? printableCharacters.length / text.length
        : 0;

    return printableRatio > 0.85;
}

// it runs when a file has been read successfully
// puts the extracted text into the main text area 
function finishFileProcessing(extractedText, statusMessage) {
    text.value = extractedText;
    inputStatus.textContent = statusMessage;
    fileProcessing = false;
    extractButton.disabled = false;
}

function setActiveStep(step) {
    const steps = document.querySelectorAll('.workflow-steps span');
    steps.forEach((item) => item.classList.remove('active'));

    const activeStep = document.querySelector('[data-step="' + step + '"]');
    if (activeStep) {
        activeStep.classList.add('active');
    }
}

//it reads the input text and if there's any text then the NER pipeline section is shown
//if its empty then an error message will be shown 
extractButton.addEventListener('click', () => {
    if (fileProcessing) {
        inputStatus.textContent = 'Please wait while the selected file is being processed.';
        return;
    }

    const textValue = text.value.trim();
    if (textValue) {
        pipelineStage.classList.remove('d-none');
        setActiveStep('pipeline');
    } else {
        alert("Please enter some text to extract.")
    }
});

// it clears the text area and file picker
// it hides every section using bootstrap's d-none class 
clearButton.addEventListener('click', () => {
    text.value = '';
    fileInput.value = '';
    fileProcessing = false;
    extractButton.disabled = false;
    inputStatus.textContent = '';
    pipelineStage.classList.add('d-none');
    reviewStage.classList.add('d-none');
    editStage.classList.add('d-none');
    outputStage.classList.add('d-none');
    resultsBody.innerHTML = '<tr><td colspan="3" class="text-muted">Waiting for NER API results.</td></tr>';
    outputText.value = '';
    entities = [];
    setActiveStep('input');
});

// it runs when the user selects a file
fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    fileProcessing = true;
    extractButton.disabled = true;
    inputStatus.textContent = 'Reading ' + file.name + '...';

    if (file.name.endsWith('.txt')){
        const reader = new FileReader();

        reader.addEventListener('load', () => {
        finishFileProcessing(reader.result, 'TXT text is ready.');
        });

    reader.readAsText(file);
    }
    else if (file.name.endsWith('.pdf')){
       const reader = new FileReader();

       reader.onload = async function() {
        try {
        const pdf = await pdfjsLib.getDocument({data: reader.result, disableWorker: true}).promise;

        let result = '';

        for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
            inputStatus.textContent = 'Reading PDF page ' + pageNumber + ' of ' + pdf.numPages + '...';
            const page = await pdf.getPage(pageNumber);
            const content = await page.getTextContent();

            for (const item of content.items) {
                result += item.str + ' ';
            }

            result += '\n';

        }

        if (hasReadableText(result)) {
            finishFileProcessing(result, 'PDF text is ready.');
            return;
        }

        text.value = '';
        for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
            inputStatus.textContent = 'OCR page ' + pageNumber + ' of ' + pdf.numPages + '...';
            const page = await pdf.getPage(pageNumber);
            const viewport = page.getViewport({ scale: 2.0 });
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            canvas.width = viewport.width;
            canvas.height = viewport.height;

            await page.render({ canvasContext: context, viewport: viewport }).promise;

            const ocr = await Tesseract.recognize(canvas, 'eng+msa', {
                tessedit_pageseg_mode: '6'
            });
            text.value += ocr.data.text + '\n';
        }
        finishFileProcessing(text.value, 'Scanned PDF text is ready.');
        } catch (error) {
            console.error(error);
            inputStatus.textContent = 'Unable to read this PDF.';
            fileProcessing = false;
            extractButton.disabled = false;
        }
       };
       
       reader.readAsArrayBuffer(file);
    }   
    else if (file.name.endsWith('.docx')){
        const reader = new FileReader();

        reader.onload = async function () {
            try {
                const result = await mammoth.extractRawText({
                    arrayBuffer: reader.result
                });

                finishFileProcessing(result.value, 'DOCX text is ready.');
            } catch (error) {
                console.error(error);
                inputStatus.textContent = 'Unable to read this DOCX file.';
            }
            fileProcessing = false;
            extractButton.disabled = false;
        };

        reader.readAsArrayBuffer(file);
    }
    else if (file.name.endsWith('.jpg') || file.name.endsWith('.jpeg') ||file.name.endsWith('.png')){
        inputStatus.textContent = 'Reading text from the image...';
        try {
            const result = await Tesseract.recognize(file, 'eng+msa', {
                tessedit_pageseg_mode: '6'
            });
            finishFileProcessing(result.data.text, 'Image text is ready.');
        } catch (error) {
            console.error(error);
            inputStatus.textContent = 'Unable to read text from this image.';
        }
        fileProcessing = false;
        extractButton.disabled = false;
    }

    else {
        alert("Please select a valid file.");
        inputStatus.textContent = 'Please select a supported file.';
        fileProcessing = false;
        extractButton.disabled = false;
        return;
    }

})

sendApiButton.addEventListener('click', async () => {
    const textValue = text.value.trim();

    if (!textValue) {
        pipelineStatus.textContent = 'Please enter or select text first.';
        return;
    }
    
    pipelineStatus.textContent = 'Sending text to the NER API...';

    try {
        const response = await fetch('/api/extract', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: textValue
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'NER request failed.');
        }

        pipelineStatus.textContent = 'NER API results received.';
        reviewStage.classList.remove('d-none');
        setActiveStep('review');
        entities = data.entities;
        originalText = textValue;

        if (data.entities.length === 0) {
            resultsBody.innerHTML = '<tr><td colspan="3" class="text-muted">No entities found.</td></tr>';
            return;
        }

        resultsBody.innerHTML = '';

        for (const entity of data.entities) {
            const row = document.createElement('tr');
            const wordCell = document.createElement('td');
            const typeCell = document.createElement('td');
            const statusCell = document.createElement('td');

            wordCell.textContent = entity.entity;
            typeCell.textContent = entity.type;
            statusCell.textContent = entity.score;
            row.append(wordCell, typeCell, statusCell);
            resultsBody.appendChild(row);
        }
    } catch (error) {
        pipelineStatus.textContent = error.message;
    }
});

async function generateOutput(style) {
    if (!originalText) return;

    try {
        const response = await fetch('/api/redact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: originalText,
                style: style
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || data.detail || 'Redaction failed.');
        }

        outputText.value = data.redacted_text;
        entities = data.entities;
    } catch (error) {
        outputStatus.textContent = error.message;
    }
}

correctionYes.addEventListener('click', () => {
    editText.value = text.value;
    editStage.classList.remove('d-none');
    setActiveStep('output');
});

correctionNo.addEventListener('click', () => {
    outputStage.classList.remove('d-none');
    setActiveStep('output');
    generateOutput(labelMode.checked ? 'label' : 'mask');
});

saveEditButton.addEventListener('click', () => {
    text.value = editText.value;
    originalText = editText.value;
    editStage.classList.add('d-none');
    outputStage.classList.remove('d-none');
    setActiveStep('output');
    generateOutput(labelMode.checked ? 'label' : 'mask');
});

labelMode.addEventListener('change', () => {
    if (!outputStage.classList.contains('d-none')) {
        generateOutput('label');
    }
});

blackMode.addEventListener('change', () => {
    if (!outputStage.classList.contains('d-none')) {
        generateOutput('mask');
    }
});

copyButton.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(outputText.value);
        outputStatus.textContent = 'Output copied.';
    } catch (error) {
        outputStatus.textContent = 'Unable to copy output.';
    }
});

downloadButton.addEventListener('click', () => {
    const file = new Blob([outputText.value], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(file);
    link.download = 'redacted-output.txt';
    link.click();
    URL.revokeObjectURL(link.href);
    outputStatus.textContent = 'Download started.';
});