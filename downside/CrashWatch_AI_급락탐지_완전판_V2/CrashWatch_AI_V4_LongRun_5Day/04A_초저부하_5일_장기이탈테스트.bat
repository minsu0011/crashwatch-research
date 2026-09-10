@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 04A_5일_장기이탈테스트.py --config configs\longrun_5day_cool.json
pause
