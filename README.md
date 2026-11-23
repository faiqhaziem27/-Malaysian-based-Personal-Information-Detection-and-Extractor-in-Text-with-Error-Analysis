# Malaysian Personal Information Detector

A local web app that detects and redacts Malaysian personal information in text.

Open the app at <http://127.0.0.1:5000>.

## Download

### Download as ZIP

1. Open this project on GitHub.
2. Click **Code**.
3. Click **Download ZIP**.
4. Extract the downloaded ZIP file.

### Or use Git

```powershell
git clone https://github.com/faiqhaziem27/-Malaysian-based-Personal-Information-Detection-and-Extractor-in-Text-with-Error-Analysis.git
cd -Malaysian-based-Personal-Information-Detection-and-Extractor-in-Text-with-Error-Analysis
```

## Requirements

- Windows
- Python 3.12
- Internet connection

## First-time setup

Open PowerShell inside the extracted project folder.

1. Install Python 3.12 if needed:

   ```powershell
   winget install --id Python.Python.3.12 --exact
   ```

2. Create the Python environment:

   ```powershell
   py -3.12 -m venv .venv
   ```

3. Allow the environment to activate:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

4. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. Install the required packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

The setup may take a few minutes.

## Run the app

The app needs two PowerShell windows.

### Window 1: Start the backend

```powershell
cd C:\path\to\the\project
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Replace `C:\path\to\the\project` with the real folder location.

The first startup may download and load the NER model. Wait until you see:

```text
Application startup complete.
```

Do not close this window.

### Window 2: Start the website

```powershell
cd C:\path\to\the\project
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python app.py
```

Do not close this window either.

Open <http://127.0.0.1:5000> in your browser.

## Stop the app

Press `Ctrl+C` in both PowerShell windows.

## Restart the app

After the setup is complete, use the two commands above in two separate PowerShell windows.

## If port 8000 is busy

If the backend is already running, open <http://127.0.0.1:8000/health>.

If it is not running, start it on port `8001`:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
```

Then start the website with:

```powershell
$env:FASTAPI_URL="http://127.0.0.1:8001"
python app.py
```

## Common problems

### PowerShell says activation is disabled

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Python is not found

Install Python 3.12:

```powershell
winget install --id Python.Python.3.12 --exact
```
