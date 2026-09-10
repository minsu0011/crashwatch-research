@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .env (
  copy .env.example .env >nul
  echo .env 파일을 만들었습니다. DATA_GO_KR_SERVICE_KEY와 ECOS_API_KEY를 입력한 뒤 다시 실행하세요.
  notepad .env
  pause
  exit /b 1
)
python -m pip install -r requirements_priority_data.txt
python priority_data_downloader.py --sources opendart,data_go,ecos --start 2018-01-01 --dart-years 2018-2025 --dart-reports annual,half,q1,q3
pause
