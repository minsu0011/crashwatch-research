@echo off
setlocal
cd /d "%~dp0.."
python starter_code\run_surge_correlation_map.py --package-root "%CD%" --resume
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] Surge correlation map failed with exit code %EXIT_CODE%.
) else (
  echo.
  echo [OK] outputs\surge_correlation_map has been generated.
)
pause
exit /b %EXIT_CODE%
