@echo off
cd /d "%~dp0"
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
python run_normal_market_experiment.py --project-root "%CD%\.." %*
pause
