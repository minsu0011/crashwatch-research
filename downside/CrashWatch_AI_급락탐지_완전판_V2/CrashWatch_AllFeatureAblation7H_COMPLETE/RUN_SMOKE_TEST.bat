@echo off
setlocal
cd /d "%~dp0"
python smoke_test.py
if errorlevel 1 pause
