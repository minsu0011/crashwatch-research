@echo off
setlocal
set "PKG=%~dp0"
set "PY=%PKG%..\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Python virtual environment not found: %PY%
  exit /b 1
)
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=32
set LIGHTGBM_NUM_THREADS=32
"%PY%" "%PKG%run_regime_expert_router_full_load.py" --total-threads 32
exit /b %ERRORLEVEL%
