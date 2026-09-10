@echo off
setlocal EnableExtensions
cd /d "%~dp0"
python -m py_compile run_surge_competingrisk_hardfp_v14.py surge_v14_models.py run_surge_magnitude_direction_v13.py surge_v13_data.py surge_v13_models.py verify_surge_competingrisk_hardfp_v14.py synthetic_e2e_v14.py test_surge_competingrisk_hardfp_v14.py
if errorlevel 1 exit /b 1
python -m unittest -v test_surge_competingrisk_hardfp_v14.py
if errorlevel 1 exit /b 1
python synthetic_e2e_v14.py
if errorlevel 1 exit /b 1
echo V14 CHECKS PASS
endlocal
