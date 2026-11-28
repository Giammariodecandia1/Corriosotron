$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Ambiente .venv assente: eseguire installa_windows.bat.' }
$listener = Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
    if ($owner.ExecutablePath -notlike "$project\*" -or $owner.CommandLine -notlike '*uvicorn main:app*') {
        throw "La porta 8767 e' occupata da un processo esterno alla v3 (PID $($listener[0].OwningProcess))."
    }
    $backendTime = (Get-ChildItem -LiteralPath $project -Filter '*.py' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    if ($owner.CreationDate -ge $backendTime) { Write-Host 'v3 aggiornata gia in ascolto su http://127.0.0.1:8767'; exit 0 }
    Write-Host 'Backend aggiornato: riavvio solo i processi Uvicorn di questa v3.'
    $processes = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.ExecutablePath -like "$project\*" -and $_.CommandLine -like '*uvicorn main:app --host 127.0.0.1 --port 8767*' }
    foreach ($item in $processes) { Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($attempt=0; $attempt -lt 10 -and (Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
    if (Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue) {
        throw 'La porta 8767 non si e liberata dopo il riavvio della v3.'
    }
}
$env:HF_HOME = Join-Path $project 'model_cache'
$env:DINO_CPU_THREADS = '4'
$env:POC_HOST = '127.0.0.1'
$env:POC_PORT = '8767'
$out = Join-Path $project 'server.log'
$err = Join-Path $project 'server-error.log'
$proc = Start-Process -FilePath $python -ArgumentList '-m uvicorn main:app --host 127.0.0.1 --port 8767' -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
# ultralytics makes the import slower than the first tool: wait up to 40 s.
for ($attempt=0; $attempt -lt 40 -and -not (Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue); $attempt++) {
    Start-Sleep -Seconds 1
}
if (-not (Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue)) { throw "Avvio fallito; controllare $err" }
Write-Host "v3 avviata (PID $($proc.Id)) su http://127.0.0.1:8767"
