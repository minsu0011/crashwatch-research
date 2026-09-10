@echo off
setlocal
cd /d "%~dp0"
python -m py_compile surge_v13_data.py surge_v13_models.py run_surge_magnitude_direction_v13.py verify_surge_magnitude_direction_v13.py synthetic_e2e_v13.py test_surge_magnitude_direction_v13.py
if errorlevel 1 exit /b 1
python -m unittest -v test_surge_magnitude_direction_v13.py
if errorlevel 1 exit /b 1
if exist _synthetic_check rmdir /s /q _synthetic_check
python synthetic_e2e_v13.py --output _synthetic_check
if errorlevel 1 exit /b 1
echo V13 ALL CHECKS PASS
endlocal
