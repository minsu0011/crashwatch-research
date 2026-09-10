@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 03C_이원화_이탈테스트.py %*
pause
