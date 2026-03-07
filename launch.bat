@echo off
title Duplicate File Finder Launcher
set "ENV_PY=%~dp0.venv-app\Scripts\python.exe"
echo.
echo ====================================
echo  DUPLICATE FILE FINDER LAUNCHER
echo ====================================
echo.
if exist "%ENV_PY%" (
    echo Using isolated environment: .venv-app
    "%ENV_PY%" launcher.py
) else (
    echo Using system Python. For a clean app-only environment run:
    echo   powershell -ExecutionPolicy Bypass -File setup_app_env.ps1
    python launcher.py
)
pause
