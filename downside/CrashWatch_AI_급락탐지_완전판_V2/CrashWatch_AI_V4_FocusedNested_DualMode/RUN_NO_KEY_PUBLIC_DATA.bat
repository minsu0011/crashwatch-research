@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv_data\Scripts\python.exe (
  py -3.11 -m venv .venv_data 2>nul || python -m venv .venv_data
  .venv_data\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
  .venv_data\Scripts\python.exe -m pip install -r requirements_required_data.txt
)
.venv_data\Scripts\python.exe 06A_필수데이터_통합수집.py --start 2018-01-01 --end auto --sources macro,naver --no-dart-documents
endlocal
pause
