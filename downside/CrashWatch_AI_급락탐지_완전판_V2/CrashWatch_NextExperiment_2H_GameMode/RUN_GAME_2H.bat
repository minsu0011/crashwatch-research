@echo off
chcp 65001 > nul
cd /d "%~dp0"
set OMP_NUM_THREADS=3
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set CUDA_DEVICE_MAX_CONNECTIONS=1
python run_next_experiment_2h.py %*
pause
