@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set "FINANCE_PYTHON=..\.venv\Scripts\python.exe"
if not exist "%FINANCE_PYTHON%" set "FINANCE_PYTHON=python"
set OMP_NUM_THREADS=16
set MKL_NUM_THREADS=16
set OPENBLAS_NUM_THREADS=16
set NUMEXPR_NUM_THREADS=16
set NUMBA_NUM_THREADS=16
set CUDA_VISIBLE_DEVICES=0
set PYTHONUNBUFFERED=1
set NUMBA_CACHE_DIR=%CD%\.numba_cache
powercfg /change standby-timeout-ac 0 > nul 2>&1
powercfg /change monitor-timeout-ac 0 > nul 2>&1
"%FINANCE_PYTHON%" 04Z_11시간_금융전체실행.py --hours 11 --threads 16 --folds 8 --seeds 17,43 --cache-namespace finance11h_v2
pause
