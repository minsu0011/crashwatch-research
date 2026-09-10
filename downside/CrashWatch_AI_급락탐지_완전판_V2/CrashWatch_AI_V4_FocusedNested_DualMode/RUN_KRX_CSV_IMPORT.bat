@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv_data\Scripts\python.exe call SETUP_REQUIRED_DATA.bat
.venv_data\Scripts\python.exe 06C_KRX_CSV_가져오기.py external_inputs\licensed_krx_csv_inbox --ratio-unit percent
set EXIT_CODE=%ERRORLEVEL%
echo.
if not %EXIT_CODE%==0 echo CSV 형식 또는 컬럼 인식 실패입니다. licensed_csv_import_manifest.csv를 확인하세요.
endlocal
pause
