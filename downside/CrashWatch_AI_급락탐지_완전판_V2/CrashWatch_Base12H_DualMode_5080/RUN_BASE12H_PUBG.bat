@echo off
setlocal
cd /d "%~dp0"
set "HOURS=%~1"
if "%HOURS%"=="" set "HOURS=12"
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4
set NUMBA_NUM_THREADS=4
set PYTHONHASHSEED=0
python run_base12h.py --profile pubg --hours %HOURS% --workers 1
set "EXITCODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
