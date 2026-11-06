# -Malaysian-based-Personal-Information-Detection-and-Extractor-in-Text-with-Error-Analysis

## Run the frontend locally

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
2. Open PowerShell in this project folder.
3. Run:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python app.py
   ```

4. Open <http://127.0.0.1:5000> in your browser.

This Flask app serves the frontend only. OCR, NER detection, replacement generation, and production file processing are intentionally left for the backend integration.

On Windows, you can also double-click `run.bat` after installing Python.