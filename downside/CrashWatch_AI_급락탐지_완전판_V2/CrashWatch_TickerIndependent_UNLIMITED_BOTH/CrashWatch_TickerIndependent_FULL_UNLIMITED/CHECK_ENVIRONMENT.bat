@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "CRASHWATCH_EXECUTION_PROFILE=full"
set "PYTHONUNBUFFERED=1"
set "CUDA_VISIBLE_DEVICES=0"
set "OMP_NUM_THREADS=16"
set "MKL_NUM_THREADS=16"
set "OPENBLAS_NUM_THREADS=16"
set "NUMEXPR_NUM_THREADS=16"
echo Checking FULL CPU/GPU stack...
python check_full_stack.py
pause
