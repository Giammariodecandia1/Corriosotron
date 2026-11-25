$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Ambiente .venv assente: eseguire installa_windows.bat.' }
$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
    if ($owner.ExecutablePath -notlike "$project\*" -or $owner.CommandLine -notlike '*uvicorn main:app*') {
        throw "La porta 8765 e' occupata da un processo esterno alla PoC (PID $($listener[0].OwningProcess))."
    }
    $backendTime = (Get-ChildItem -LiteralPath $project -Filter '*.py' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    if ($owner.CreationDate -ge $backendTime) { Write-Host 'PoC aggiornata gia in ascolto su http://127.0.0.1:8765'; exit 0 }
    Write-Host 'Backend aggiornato: riavvio solo i processi Uvicorn di questa PoC.'
    $processes = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.ExecutablePath -like "$project\*" -and $_.CommandLine -like '*uvicorn main:app --host 127.0.0.1 --port 8765*' }
    foreach ($item in $processes) { Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($attempt=0; $attempt -lt 10 -and (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
    if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) {
        throw 'La porta 8765 non si e liberata dopo il riavvio della PoC.'
    }
}
$env:HF_HOME = Join-Path $project 'model_cache'
$env:DINO_CPU_THREADS = '4'
$env:POC_HOST = '127.0.0.1'
$env:POC_PORT = '8765'
$out = Join-Path $project 'server.log'
$err = Join-Path $project 'server-error.log'
$proc = Start-Process -FilePath $python -ArgumentList '-m uvicorn main:app --host 127.0.0.1 --port 8765' -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
Start-Sleep -Seconds 2
if (-not (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)) { throw "Avvio fallito; controllare $err" }
Write-Host "PoC avviata (PID $($proc.Id)) su http://127.0.0.1:8765"
