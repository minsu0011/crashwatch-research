@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 01B_이원화_데이터_크롤링.py %*
pause
