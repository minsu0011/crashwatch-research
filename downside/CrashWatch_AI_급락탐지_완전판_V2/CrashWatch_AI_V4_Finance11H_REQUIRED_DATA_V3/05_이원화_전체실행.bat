@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 05_이원화_전체실행.py %*
pause
