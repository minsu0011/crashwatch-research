@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv_data\Scripts\python.exe call SETUP_REQUIRED_DATA.bat
.venv_data\Scripts\python.exe 06A_필수데이터_통합수집.py --start 2018-01-01 --end auto --sources macro,krx,lending,dart,naver --krx-chunk-months 24 --lending-chunk-months 12 --strict
endlocal
pause
