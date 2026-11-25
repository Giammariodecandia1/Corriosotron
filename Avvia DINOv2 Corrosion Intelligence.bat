@echo off
setlocal
set "POC_DIR=%~dp0dinov2-corrosion-poc"
if not exist "%POC_DIR%\avvia_server.ps1" (
  echo Non trovo la cartella dinov2-corrosion-poc accanto a questo file.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%POC_DIR%\avvia_server.ps1"
if errorlevel 1 (
  echo Avvio non riuscito. Controlla server-error.log nella cartella dinov2-corrosion-poc.
  pause
  exit /b 1
)
for /l %%I in (1,1,20) do (
  powershell.exe -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765 -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
  if not errorlevel 1 goto open
  timeout /t 1 /nobreak >nul
)
echo Il server non risponde. Controlla i file server-error.log nella cartella dinov2-corrosion-poc.
pause
exit /b 1
:open
start "" "http://127.0.0.1:8765"
exit /b 0
