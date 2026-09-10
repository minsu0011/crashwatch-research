@echo off
chcp 65001 > nul
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"
pause
