@echo off
cd /d "%~dp0"
set "CUDA_VISIBLE_DEVICES=0"
python check_gpu_stack.py
pause
