@echo off
setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=8
python starter_code\run_surge_leadlag_network_v11.py --package-root . --all-tickers --threads 8 --rolling-step 20 --run-probe --resume
if errorlevel 1 exit /b 1
python starter_code\verify_surge_leadlag_network_v11.py --output outputs\surge_leadlag_network_v11
