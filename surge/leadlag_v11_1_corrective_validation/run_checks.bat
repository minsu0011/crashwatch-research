@echo off
setlocal
cd /d "%~dp0"
python -m py_compile surge_leadlag_common_v11_1.py surge_leadlag_corrective_v11_1.py run_surge_leadlag_v11_1.py verify_surge_leadlag_v11_1.py test_surge_leadlag_v11_1.py synthetic_e2e_v11_1.py
if errorlevel 1 exit /b 1
python -m unittest -v test_surge_leadlag_v11_1.py
if errorlevel 1 exit /b 1
python synthetic_e2e_v11_1.py
if errorlevel 1 exit /b 1
echo ALL_CHECKS_PASS
endlocal
