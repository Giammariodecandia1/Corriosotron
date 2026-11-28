@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Esegui prima installa_windows.bat & pause & exit /b 1)
set "HF_HOME=%~dp0model_cache"
set "DINO_CPU_THREADS=4"
echo Browser della VM: http://127.0.0.1:8767
.venv\Scripts\python.exe main.py
pause
