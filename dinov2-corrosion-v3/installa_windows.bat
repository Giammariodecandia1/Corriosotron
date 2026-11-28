@echo off
setlocal
cd /d "%~dp0"
rem v2: same base as the first tool plus ultralytics (YOLO + SAM) for object recognition.
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
if not exist model_cache\seg_env\Scripts\python.exe (
  if not exist model_cache mkdir model_cache
  py -3.11 -m venv model_cache\seg_env
  if errorlevel 1 goto fail
)
model_cache\seg_env\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
model_cache\seg_env\Scripts\python.exe -m pip install "rembg[cpu]==2.0.87"
if errorlevel 1 goto fail
rem Download the SAM checkpoint once (model_cache\sam).
.venv\Scripts\python.exe -c "import object_detection as od; print(od.ensure_sam())"
if errorlevel 1 goto fail
echo Installazione completata.
pause
exit /b 0
:fail
echo ERRORE installazione. Richiesto Python 3.11 x64.
pause
exit /b 1
