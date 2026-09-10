@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
python run_surge_leadlag_v11_1.py --threads 16 --maxstat-executor process
if errorlevel 1 (
  echo.
  echo V11.1 FAILED. Check outputs\surge_leadlag_v11_1_corrective_validation\RUN_STATUS.json
  exit /b 1
)
echo.
echo V11.1 COMPLETE.
endlocal
