@echo off
setlocal
title Rustron - Corrosion Intelligence - server (NON chiudere questa finestra)
cd /d "%~dp0"

echo.
echo  ==========================================================
echo   Rustron / Corrosion Intelligence
echo   Indirizzo: http://127.0.0.1:8766
echo  ==========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo  [ERRORE] Ambiente Python della v2 non trovato in:
  echo           %~dp0.venv
  echo  Eseguire prima installa_windows.bat
  echo.
  pause
  exit /b 1
)

echo  [1/3] Chiudo un eventuale server v2 gia' aperto (un'analisi in corso verrebbe interrotta)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root=(Resolve-Path '.').Path; $c=Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue;" ^
  "foreach($x in $c){ $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$x.OwningProcess);" ^
  " if($p.ExecutablePath -notlike ($root+'\*')){ Write-Host ('        [ERRORE] porta 8766 occupata da un altro programma: '+$p.Name); exit 1 } }" ^
  "$old=@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.ExecutablePath -like ($root+'\*') });" ^
  "foreach($p in $old){ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }" ^
  "if($old.Count){ Write-Host ('        chiusi '+$old.Count+' processi della v2 rimasti aperti') }"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)

set "HF_HOME=%~dp0model_cache"
set "DINO_CPU_THREADS=4"
set "POC_HOST=127.0.0.1"
set "POC_PORT=8766"

echo  [2/3] Il browser si aprira' da solo quando il server e' pronto (circa 10-30 secondi).
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command ^
  "for($i=0;$i -lt 120;$i++){ if(Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue){ Start-Process 'http://127.0.0.1:8766'; break }; Start-Sleep -Milliseconds 500 }"

echo  [3/3] Avvio del server. Per spegnerlo chiudere questa finestra.
echo.
".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8766

echo.
echo  Il server si e' fermato. Se c'e' un errore qui sopra, inviarne una foto.
pause
