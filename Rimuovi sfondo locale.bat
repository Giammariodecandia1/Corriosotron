@echo off
setlocal
cd /d "%~dp0dinov2-corrosion-poc"
set "REMBG_HOME=%CD%\model_cache\rembg"
set "PYTHON_EXE=%CD%\model_cache\seg_env\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Ambiente per la rimozione dello sfondo non trovato.
    echo Contattare il responsabile della PoC per completare l'installazione locale.
    pause
    exit /b 1
)
"%PYTHON_EXE%" rimuovi_sfondo_locale.py %*
if errorlevel 1 (
    echo.
    echo Elaborazione non completata. Leggere il messaggio sopra.
)
pause
