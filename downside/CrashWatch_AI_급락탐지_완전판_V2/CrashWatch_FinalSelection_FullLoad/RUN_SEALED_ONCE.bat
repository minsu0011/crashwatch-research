@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set CUDA_VISIBLE_DEVICES=0
python run_sealed_once.py %*
set EXITCODE=%ERRORLEVEL%
pause
exit /b %EXITCODE%
