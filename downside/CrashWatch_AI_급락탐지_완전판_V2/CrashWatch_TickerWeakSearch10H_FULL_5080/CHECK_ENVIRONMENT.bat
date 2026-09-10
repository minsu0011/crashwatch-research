@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
set "CRASHWATCH_EXECUTION_PROFILE=full"
set "PYTHONUNBUFFERED=1"
set "CUDA_VISIBLE_DEVICES=0"
echo Checking XGBoost CUDA, CatBoost CUDA, and LightGBM CPU...
python check_full_stack.py
pause
