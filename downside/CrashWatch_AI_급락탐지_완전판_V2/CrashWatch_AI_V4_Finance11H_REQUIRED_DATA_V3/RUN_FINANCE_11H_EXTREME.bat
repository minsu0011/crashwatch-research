@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv_data\Scripts\python.exe call SETUP_REQUIRED_DATA.bat
set OMP_NUM_THREADS=16
set MKL_NUM_THREADS=16
set OPENBLAS_NUM_THREADS=16
set NUMEXPR_NUM_THREADS=16
set NUMBA_NUM_THREADS=16
set CUDA_VISIBLE_DEVICES=0
set PYTHONUNBUFFERED=1
set NUMBA_CACHE_DIR=%CD%\.numba_cache
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-ac 0 >nul 2>&1
.venv_data\Scripts\python.exe 04Z_11시간_금융전체실행.py --hours 11 --threads 16 --folds 8 --seeds 17,43 --cache-namespace finance11h_v3 --skip-data-download
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
if not %EXIT_CODE%==0 echo CHECK_REQUIRED_DATA.bat에서 차단 원인을 확인하세요.
endlocal
pause
