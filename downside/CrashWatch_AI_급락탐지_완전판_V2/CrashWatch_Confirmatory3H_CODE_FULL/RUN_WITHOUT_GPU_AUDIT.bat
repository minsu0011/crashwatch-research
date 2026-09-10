@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0.."
if exist "%~dp0crashwatch_ai_data\development\training_dataset_finance11h.parquet" set "PROJECT_ROOT=%~dp0"
if exist "%~dp0..\crashwatch_ai_data\development\training_dataset_finance11h.parquet" set "PROJECT_ROOT=%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
if "%~1"=="" (
  python run_confirmatory_3h.py all --project-root "%PROJECT_ROOT%" --no-gpu-audit
) else (
  python run_confirmatory_3h.py all --project-root "%PROJECT_ROOT%" --dataset "%~1" --no-gpu-audit
)
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
