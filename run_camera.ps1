param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$ScriptPath = ".\receipt_scanner.py",
    [string]$DatabasePath = ".\receipts.db",
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

Write-Host "Starting receipt scanner (camera mode)..." -ForegroundColor Cyan

if (-not (Test-Path $PythonExe)) {
    Write-Host "Local venv python not found at $PythonExe" -ForegroundColor Yellow
    Write-Host "Run setup first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python -m pip install opencv-python numpy pytesseract flask werkzeug" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Host "Scanner script not found: $ScriptPath" -ForegroundColor Red
    exit 1
}

try {
    & tesseract --version | Out-Null
}
catch {
    Write-Host "Tesseract is not available in PATH." -ForegroundColor Red
    Write-Host "Install Tesseract OCR and ensure tesseract.exe is on PATH." -ForegroundColor Red
    exit 1
}

$argsList = @($ScriptPath, "--camera", "--db", $DatabasePath)
if ($Debug) {
    $argsList += "--debug"
}

Write-Host "Launching camera scanner..." -ForegroundColor Green
& $PythonExe @argsList
