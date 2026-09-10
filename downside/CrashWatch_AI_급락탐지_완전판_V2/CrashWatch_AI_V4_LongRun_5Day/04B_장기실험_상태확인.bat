@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 04B_장기실험_상태확인.py
pause
