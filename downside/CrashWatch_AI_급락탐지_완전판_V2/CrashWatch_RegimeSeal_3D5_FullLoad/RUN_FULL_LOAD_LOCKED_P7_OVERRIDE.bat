@echo off
setlocal
set "PKG=%~dp0"
set "PY=%PKG%..\.venv\Scripts\python.exe"
if not exist "%PY%" exit /b 1
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=32
"%PY%" "%PKG%run_regime_router_locked_override_full_load.py" --total-threads 32 --bootstrap-reps 1000 --bootstrap-workers 12
exit /b %ERRORLEVEL%
