@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "CRASHWATCH_EXECUTION_PROFILE=full"
set "PYTHONUNBUFFERED=1"
set "CUDA_VISIBLE_DEVICES=0"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "OPENBLAS_NUM_THREADS=4"
set "NUMEXPR_NUM_THREADS=4"
echo ============================================================
echo CrashWatch Weak-Model Search - FULL LOAD
echo Ryzen 9800X3D / RTX 5080 / RAM 32GB target
echo Strong models are locked. Weak models run first, middle tier second.
echo Soft budget: 10 hours. Hard checkpoint cap: 12 hours.
echo Result: %%CRASHWATCH_DATA_DIR%%\ticker_weaksearch10h_full
echo ============================================================
python run_ticker_elite2h.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXIT_CODE%
echo Result folder: %%CRASHWATCH_DATA_DIR%%\ticker_weaksearch10h_full
pause
exit /b %EXIT_CODE%
