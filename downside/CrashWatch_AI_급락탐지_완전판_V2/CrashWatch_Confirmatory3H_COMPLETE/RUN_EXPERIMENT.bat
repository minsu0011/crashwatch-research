@echo off
chcp 65001 >nul
cd /d "%~dp0"
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
python run_confirmatory_3h.py --config config.json
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
