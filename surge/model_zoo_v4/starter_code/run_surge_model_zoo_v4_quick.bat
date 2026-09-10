@echo off
setlocal
cd /d "%~dp0\.."
python starter_code\run_surge_model_zoo_v4.py ^
  --package-root . ^
  --quick ^
  --screen-seeds 17 ^
  --final-seeds 17,29,41 ^
  --finalist-count 5 ^
  --device auto ^
  --allow-cpu-fallback ^
  --threads-per-model 6 ^
  --xgboost-threads 6 ^
  --resume
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo 실행 실패. outputs\surge_model_zoo_v4\RUN_STATUS.json을 확인하십시오.
if "%EXITCODE%"=="0" echo 실행 완료. outputs\surge_model_zoo_v4를 확인하십시오.
pause
exit /b %EXITCODE%
