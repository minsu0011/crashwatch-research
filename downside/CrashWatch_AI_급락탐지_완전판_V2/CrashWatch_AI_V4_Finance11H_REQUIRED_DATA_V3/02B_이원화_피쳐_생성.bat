@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 02B_이원화_피쳐_생성.py %*
pause
