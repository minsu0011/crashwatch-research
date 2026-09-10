@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=4
set LIGHTGBM_NUM_THREADS=4
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
python run_full_selection.py %*
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
