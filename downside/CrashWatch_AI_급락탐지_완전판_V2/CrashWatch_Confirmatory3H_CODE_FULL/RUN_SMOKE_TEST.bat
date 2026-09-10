@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
python tests\smoke_test.py
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Smoke-test exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
