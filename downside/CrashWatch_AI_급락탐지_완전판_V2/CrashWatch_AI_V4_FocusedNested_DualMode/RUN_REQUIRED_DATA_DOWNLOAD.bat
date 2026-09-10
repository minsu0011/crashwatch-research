@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv_data\Scripts\python.exe call SETUP_REQUIRED_DATA.bat
.venv_data\Scripts\python.exe 06A_필수데이터_통합수집.py --start 2018-01-01 --end auto --sources macro,krx,lending,dart,naver --krx-chunk-months 24 --lending-chunk-months 12 --strict
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Exit code: %EXIT_CODE%
if not %EXIT_CODE%==0 echo 일부 데이터는 저장됐습니다. CHECK_REQUIRED_DATA.bat 확인 후 RESUME_REQUIRED_DATA_DOWNLOAD.bat으로 재개하세요.
endlocal
pause
