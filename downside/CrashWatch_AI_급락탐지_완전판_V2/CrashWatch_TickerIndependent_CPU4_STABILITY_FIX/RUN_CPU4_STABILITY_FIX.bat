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
echo ============================================================
echo CrashWatch Ticker Independent - CPU4 Stability Fix
echo GPU sealed. Total CPU limit: 4 logical threads.
echo No wall-clock deadline. Runs until all registered tasks finish.
echo Result: %%CRASHWATCH_DATA_DIR%%\ticker_independent_cpu4_stability_fix
echo ============================================================
python run_ticker_elite2h.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXIT_CODE%
echo Result folder: %%CRASHWATCH_DATA_DIR%%\ticker_independent_cpu4_stability_fix
pause
exit /b %EXIT_CODE%
