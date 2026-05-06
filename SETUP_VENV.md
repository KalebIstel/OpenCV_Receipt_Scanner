# Receipt Scanner - Git Bash venv Setup (No Docker)

This guide is for running the project from a **Git Bash** terminal on Windows.

## 1) Prerequisites

- Python 3.10+ installed and available in PATH
- Tesseract OCR installed and available in PATH (`tesseract --version`)
- A webcam (optional, for camera mode)

## 2) Open project in Git Bash

From Git Bash:

```bash
cd /d/Coding/OpenCV/OpenCV_Receipt_Scanner
```

## 3) Create and activate virtual environment

From inside the project folder:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When activated, your prompt should show `(.venv)`.

## 4) Verify installation

```bash
python -c "import cv2, numpy, pytesseract, flask; print('OK')"
tesseract --version
```

## 5) Run examples (Git Bash)

### A) Live camera view + manual capture

Shows real-time camera preview and scanner preview windows.  
Press **Enter** or **Space** to capture, **Q** or **Esc** to cancel.

```bash
python receipt_scanner.py --camera --live-preview
```

### B) Live camera view + auto capture

Auto-captures when the receipt is stable and aligned.

```bash
python receipt_scanner.py --camera --auto-capture
```

Optional: tune stability requirement (default `12` frames):

```bash
python receipt_scanner.py --camera --auto-capture --stable-frames 16
```

### C) Live camera view without scanner preview window

Use this if you only want one camera window:

```bash
python receipt_scanner.py --camera --live-preview --no-scanner-preview
```

### D) Scan one receipt image file

```bash
python receipt_scanner.py sample.jpg
```

### E) Batch scan all images in a folder

```bash
python receipt_scanner.py --batch sample_receipts
```

### F) Show stats

```bash
python receipt_scanner.py --stats
```

### G) Export to CSV

```bash
python receipt_scanner.py --export receipts_export.csv
```

## 6) Inspect saved receipts in SQLite (`receipts.db`)

After scanning, receipt data is stored in `receipts.db` in the project root.

Start SQLite shell:

```bash
sqlite3 receipts.db
```

Inside SQLite:

```sql
.tables
.schema receipts
.schema receipt_items

SELECT id, timestamp, vendor, date, total, confidence
FROM receipts
ORDER BY id DESC
LIMIT 20;

SELECT receipt_id, description, price
FROM receipt_items
ORDER BY id DESC
LIMIT 30;
```

Exit SQLite shell:

```sql
.quit
```

## 7) Deactivate virtual environment

```bash
deactivate
```

## Notes

- Default SQLite database: `receipts.db` (project root).
- Processed images are saved to `scanned_receipts`.
- If Tesseract is installed but not detected, add it to PATH or set `pytesseract.pytesseract.tesseract_cmd` to full `tesseract.exe` path.
