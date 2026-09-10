@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "CRASHWATCH_EXECUTION_PROFILE=cpu4"
set "PYTHONUNBUFFERED=1"
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "OPENBLAS_NUM_THREADS=4"
set "NUMEXPR_NUM_THREADS=4"
python request_ticker_independent_stop.py
pause
