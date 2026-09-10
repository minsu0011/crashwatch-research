@echo off
setlocal
cd /d "%~dp0"
set "HOURS=%~1"
if "%HOURS%"=="" set "HOURS=12"
set "WORKERS=%~2"
if "%WORKERS%"=="" set "WORKERS=1"
set "CUDA_VISIBLE_DEVICES=0"
set "OMP_NUM_THREADS=16"
set "MKL_NUM_THREADS=16"
set "OPENBLAS_NUM_THREADS=16"
set "NUMEXPR_NUM_THREADS=16"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256"
set "TOKENIZERS_PARALLELISM=false"
python run_refine12h.py --profile full --hours %HOURS% --tree-workers %WORKERS%
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
