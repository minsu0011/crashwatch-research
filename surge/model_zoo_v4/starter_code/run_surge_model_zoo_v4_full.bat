@echo off
setlocal
cd /d "%~dp0\.."
python starter_code\run_surge_model_zoo_v4.py ^
  --package-root . ^
  --screen-seeds 17,29 ^
  --final-seeds 17,29,41,73,101 ^
  --finalist-count 8 ^
  --device auto ^
  --allow-cpu-fallback ^
  --threads-per-model 8 ^
  --xgboost-threads 8 ^
  --target-recall 0.70 ^
  --threshold-recall-buffer 0.05 ^
  --minimum-oof-precision-lift 1.05 ^
  --minimum-holdout-precision-lift 1.10 ^
  --max-alert-rate 0.65 ^
  --resume
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo 실행 실패. outputs\surge_model_zoo_v4\RUN_STATUS.json을 확인하십시오.
if "%EXITCODE%"=="0" echo 실행 완료. outputs\surge_model_zoo_v4를 확인하십시오.
pause
exit /b %EXITCODE%
