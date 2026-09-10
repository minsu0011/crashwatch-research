@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m pip install -r requirements_priority_data.txt
python priority_data_downloader.py --sources opendart --dart-years 2018-2025 --dart-reports annual,half,q1,q3
pause
