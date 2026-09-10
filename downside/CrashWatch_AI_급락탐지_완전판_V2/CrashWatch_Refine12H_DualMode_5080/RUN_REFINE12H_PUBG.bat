@echo off
setlocal
cd /d "%~dp0"
set "HOURS=%~1"
if "%HOURS%"=="" set "HOURS=12"
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "OPENBLAS_NUM_THREADS=4"
set "NUMEXPR_NUM_THREADS=4"
set "TOKENIZERS_PARALLELISM=false"
python run_refine12h.py --profile pubg --hours %HOURS%
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
