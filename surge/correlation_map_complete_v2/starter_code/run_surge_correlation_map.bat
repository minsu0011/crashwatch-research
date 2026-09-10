@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem 이 배치파일은 starter_code 바로 위 폴더를 패키지 루트로 사용한다.
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PATH에서 python.exe를 찾지 못했습니다.
    echo Conda 또는 Python 환경을 활성화한 뒤 다시 실행하십시오.
    pause
    exit /b 1
)

echo [1/2] 필수 패키지 확인...
python -c "import numpy, pandas, pyarrow, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 필수 패키지가 부족합니다.
    echo 다음 명령을 먼저 실행하십시오:
    echo python -m pip install -r starter_code\requirements_surge_correlation.txt
    pause
    exit /b 1
)

echo [2/2] CrashWatch Surge 3D5 상관관계 지도 실행...
python starter_code\run_surge_correlation_map.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [FAILED] 종료 코드: %EXIT_CODE%
    echo outputs\surge_correlation_map_complete\RUN_STATUS.json도 확인하십시오.
    pause
    exit /b %EXIT_CODE%
)

echo.
echo [SUCCESS] outputs\surge_correlation_map_complete에 결과가 생성되었습니다.
pause
exit /b 0
