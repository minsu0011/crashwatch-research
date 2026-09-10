@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements_v13.txt
if errorlevel 1 exit /b 1
python -c "import numpy,pandas,scipy,sklearn,xgboost,lightgbm,pyarrow; print('V13 requirements import PASS')"
endlocal
