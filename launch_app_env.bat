@echo off
setlocal
title Duplicate File Finder (Isolated Env)

set "ENV_PY=%~dp0.venv-app\Scripts\python.exe"

if not exist "%ENV_PY%" (
    echo.
    echo Isolated environment not found at .venv-app
    echo Run: powershell -ExecutionPolicy Bypass -File setup_app_env.ps1
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  DUPLICATE FILE FINDER (ISOLATED ENV)
echo ==========================================
echo.
"%ENV_PY%" launcher.py
pause
