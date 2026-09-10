@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
if exist ".venv_data\Scripts\python.exe" (
  set "PY=.venv_data\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
  set "PY=..\.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo [FULL MODE] RTX 5060 Ti CUDA + CPU 16스레드, High 우선순위
echo [CACHE] finance_nested_focus_v1 - GAME MODE 결과를 그대로 재사용
"%PY%" 04E_집중Nested_이탈테스트.py --profile full --hours 0 --cache-namespace finance_nested_focus_v1 --clear-stop-on-start
pause
