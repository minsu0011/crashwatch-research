@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0.."
if exist "%~dp0crashwatch_ai_data\development\training_dataset_finance11h.parquet" set "PROJECT_ROOT=%~dp0"
if exist "%~dp0..\crashwatch_ai_data\development\training_dataset_finance11h.parquet" set "PROJECT_ROOT=%~dp0.."
if "%~1"=="" (
  python run_confirmatory_3h.py aggregate --project-root "%PROJECT_ROOT%"
) else (
  python run_confirmatory_3h.py aggregate --project-root "%PROJECT_ROOT%" --dataset "%~1"
)
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
