param(
    [string]$PythonExe = "C:\Users\naman\AppData\Local\Programs\Python\Python310\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv-app"
$requirements = Join-Path $root "requirements-app-env.txt"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Write-Host "Using Python: $PythonExe"

if (-not (Test-Path $venv)) {
    & $PythonExe -m venv $venv
}

$venvPy = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "Existing .venv-app is incomplete; recreating it..."
    if (Test-Path $venv) {
        Remove-Item -Recurse -Force $venv
    }
    & $PythonExe -m venv $venv
}

Write-Host "Upgrading pip in .venv-app"
& $venvPy -m pip install --upgrade pip

Write-Host "Installing pinned app dependencies..."
& $venvPy -m pip install -r $requirements

Write-Host ""
Write-Host "Environment ready: $venv"
Write-Host "Launch with: .\.venv-app\Scripts\python.exe launcher.py"
Write-Host "Optional native dependency: install Tesseract OCR and add it to PATH for OCR fallback."
