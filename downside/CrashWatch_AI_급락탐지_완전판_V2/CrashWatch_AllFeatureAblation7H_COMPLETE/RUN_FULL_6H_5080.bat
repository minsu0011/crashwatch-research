@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_ROOT=%~dp0.."
set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=3
set LIGHTGBM_NUM_THREADS=3
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

"%PYTHON_EXE%" run_all_feature_ablation_7h.py --config config_full_6h.json --project-root "%PROJECT_ROOT%" %*
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
