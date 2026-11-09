const extractButton = document.getElementById('extract-button');
const text = document.getElementById('text');
const pipelineStage = document.getElementById('pipeline-stage');
const clearButton = document.getElementById('clear-button');
const fileInput = document.getElementById('file-input');

extractButton.addEventListener('click', () => {
    const textValue = text.value.trim();
    if (textValue) {
        pipelineStage.classList.remove('d-none');
    } else {
        alert("Please enter some text to extract.")
    }
});

clearButton.addEventListener('click', () => {
    text.value = '';
    pipelineStage.classList.add('d-none');
});

fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    if (file.name.endsWith('.txt')){
        const reader = new FileReader();

        reader.addEventListener('load', () => {
        text.value = reader.result;
        });

    reader.readAsText(file);
    }
    else if (file.name.endsWith('.pdf')){
       const reader = new FileReader();

       reader.onload = async function() {
        const pdf = await pdfjsLib.getDocument({data: reader.result, disableWorker: true}).promise;

        let result = '';

        for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
            const page = await pdf.getPage(pageNumber);
            const content = await page.getTextContent();

            for (const item of content.items) {
                result += item.str + ' ';
            }

            result += '\n';

        }

        if (result.trim()) {
            text.value = result;
            return;
        }

        text.value = '';
        for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
            const page = await pdf.getPage(pageNumber);
            const viewport = page.getViewport({ scale: 1.0});
            const canvas = document .createElement('canvas');
            const context = canvas.getContext('2d');

            canvas.width = viewport.width;
            canvas.height = viewport.height;

            await page.render({ canvasContext: context, viewport: viewport }).promise;

            const ocr = await Tesseract.recognize(canvas, 'eng');
            text.value += ocr.data.text + '\n';
        }
       };
       
       reader.readAsArrayBuffer(file);
    }   
    else if (file.name.endsWith('.docx')){
        const reader = new FileReader();

        reader.onload = async function () {
            const result = await mammoth.extractRawText({
                arrayBuffer: reader.result
            })

            text.value = result.value;
        };

        reader.readAsArrayBuffer(file);
    }
    else if (file.name.endsWith('.jpg') || file.name.endsWith('.jpeg') ||file.name.endsWith('.png')){
        const result = await Tesseract.recognize(file, 'eng');
        text.value = result.data.text;
    }

    else {
        alert("Please select a valid file.");
        return;
    }

})