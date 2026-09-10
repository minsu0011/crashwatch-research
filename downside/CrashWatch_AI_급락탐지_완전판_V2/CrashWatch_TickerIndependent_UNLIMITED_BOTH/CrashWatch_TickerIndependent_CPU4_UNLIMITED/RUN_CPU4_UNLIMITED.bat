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
echo CrashWatch Ticker Independent - CPU4
echo Completion-driven mode: no wall-clock time limit
echo Runs until every registered task completes/fails or safe stop
echo Result: %%CRASHWATCH_DATA_DIR%%\ticker_independent_cpu4_unlimited
echo ============================================================
python run_ticker_elite2h.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXIT_CODE%
echo Result folder: %%CRASHWATCH_DATA_DIR%%\ticker_independent_cpu4_unlimited
pause
exit /b %EXIT_CODE%
