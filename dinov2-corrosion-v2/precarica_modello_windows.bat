@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Installa prima & pause & exit /b 1)
set "HF_HOME=%~dp0model_cache"
.venv\Scripts\python.exe precarica_modello.py
pause
