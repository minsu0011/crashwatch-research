@echo off
setlocal
cd /d "%~dp0"
set "HOURS=%~1"
if "%HOURS%"=="" set "HOURS=12"
set "WORKERS=%~2"
if "%WORKERS%"=="" set "WORKERS=2"
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=16
set MKL_NUM_THREADS=16
set OPENBLAS_NUM_THREADS=16
set NUMEXPR_NUM_THREADS=16
set NUMBA_NUM_THREADS=16
set PYTHONHASHSEED=0
python run_base12h.py --profile full --hours %HOURS% --workers %WORKERS%
set "EXITCODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
