@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%run_camera.ps1"
set "MODE_ARG="

if not exist "%PS_SCRIPT%" (
    echo Missing PowerShell launcher: "%PS_SCRIPT%"
    exit /b 1
)

if /I "%~1"=="manual" set "MODE_ARG=-Mode manual"
if /I "%~1"=="auto" set "MODE_ARG=-Mode auto"

if "%MODE_ARG%"=="" (
    echo.
    echo Select capture mode:
    echo   [1] Manual capture (live preview; press Enter/Space to capture)
    echo   [2] Auto capture (captures when stable and aligned)
    choice /C 12 /N /M "Choose 1 or 2: "
    if errorlevel 2 (
        set "MODE_ARG=-Mode auto"
    ) else (
        set "MODE_ARG=-Mode manual"
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %MODE_ARG% %*
exit /b %ERRORLEVEL%
