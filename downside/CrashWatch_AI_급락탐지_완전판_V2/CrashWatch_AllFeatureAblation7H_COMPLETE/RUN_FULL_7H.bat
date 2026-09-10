@echo off
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=3
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
python run_all_feature_ablation_7h.py %*
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
