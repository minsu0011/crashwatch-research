@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_ROOT=%~dp0.."
set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1

if "%~1"=="" (
  "%PYTHON_EXE%" run_confirmatory_3h.py all --config confirmatory_3h_config_game_cpu4.json --project-root "%PROJECT_ROOT%" --no-gpu-audit
) else (
  "%PYTHON_EXE%" run_confirmatory_3h.py all --config confirmatory_3h_config_game_cpu4.json --project-root "%PROJECT_ROOT%" --dataset "%~1" --no-gpu-audit
)

set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
