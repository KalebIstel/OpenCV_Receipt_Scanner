param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$ScriptPath = ".\receipt_scanner.py",
    [string]$DatabasePath = ".\receipts.db",
    [ValidateSet("manual", "auto")]
    [string]$Mode,
    [int]$StableFrames = 12,
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

if (-not $Mode) {
    Write-Host ""
    Write-Host "Select capture mode:" -ForegroundColor Cyan
    Write-Host "  [1] Manual capture (live preview; press Enter/Space to capture)"
    Write-Host "  [2] Auto capture (captures when stable and aligned)"
    $selection = Read-Host "Enter 1 or 2 (default: 1)"
    if ($selection -eq "2") {
        $Mode = "auto"
    } else {
        $Mode = "manual"
    }
}

if ($Mode -eq "auto") {
    $argsList += @("--auto-capture", "--stable-frames", "$StableFrames")
    Write-Host "Launching camera scanner in AUTO mode..." -ForegroundColor Green
} else {
    $argsList += "--live-preview"
    Write-Host "Launching camera scanner in MANUAL mode..." -ForegroundColor Green
}

Write-Host "Mode: $Mode | DB: $DatabasePath" -ForegroundColor DarkGray
& $PythonExe @argsList
