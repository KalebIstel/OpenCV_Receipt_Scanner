# Receipt Scanner - venv Setup (No Docker)

This project can be moved to another laptop and run with Python virtual environment (`venv`).

## 1) Prerequisites on target laptop

- Python 3.10+ installed and available in PATH
- Tesseract OCR installed and available in PATH (`tesseract --version`)
- A webcam (optional, for camera mode)

## 2) Copy project

Copy the entire `receipt_scanner` folder to the new laptop.

## 3) Create and activate virtual environment

From inside the project folder:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows CMD

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Verify installation

```powershell
python -c "import cv2, numpy, pytesseract, flask; print('OK')"
tesseract --version
```

## 5) Run examples

### Main scanner with camera

```powershell
.\run_camera.ps1
```

Or:

```powershell
python .\receipt_scanner.py --camera
```

### Scan one receipt image

```powershell
python .\receipt_scanner.py .\sample.jpg
```

### Start advanced API

```powershell
python .\receipt_scanner_advanced.py --api
```

## Notes

- The SQLite database file defaults to `receipts.db` in the project folder.
- Scanned images are stored in `scanned_receipts`.
- If Tesseract is installed but not in PATH, either add it to PATH or set
  `pytesseract.pytesseract.tesseract_cmd` to the full `tesseract.exe` path.
