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
echo [GAME MODE] GPU 완전 비활성화, CPU 4스레드, Below Normal 우선순위
echo [CACHE] finance_nested_focus_v1 - FULL MODE와 완전 공유
"%PY%" 04E_집중Nested_이탈테스트.py --profile game --hours 0 --cache-namespace finance_nested_focus_v1 --clear-stop-on-start
pause
