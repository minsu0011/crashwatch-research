@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "CRASHWATCH_EXECUTION_PROFILE=full"
set "CUDA_VISIBLE_DEVICES=0"
set "OMP_NUM_THREADS=3"
set "MKL_NUM_THREADS=3"
set "OPENBLAS_NUM_THREADS=3"
set "NUMEXPR_NUM_THREADS=3"
python rebuild_reports_and_fingerprints.py
pause
