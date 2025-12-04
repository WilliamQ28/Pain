@echo off
REM train_yolo_gpu.bat
cd /d C:\wfc
call .venv\Scripts\activate

set NAME=wfc_ui_yolov8_coloraug2

pip install --quiet ultralytics pillow

echo Training YOLOv8 on GPU...
yolo detect train ^
  data="C:\wfc\wfc_ui.yaml" ^
  model=yolov8n.pt ^
  epochs=50 ^
  imgsz=832 ^
  batch=8 ^
  device=0 ^
  workers=2 ^
  name=%NAME%

set "BESTPT=C:\wfc\runs\detect\%NAME%\weights\best.pt"
echo.
IF EXIST "%BESTPT%" (
  echo Validating model on GPU...
  yolo detect val ^
    model="%BESTPT%" ^
    data="C:\wfc\wfc_ui.yaml" ^
    device=0
) ELSE (
  echo [WARN] best.pt not found at %BESTPT%. Skipping validation.
)

echo.
echo Done. Results under C:\wfc\runs\detect\%NAME%
pause
