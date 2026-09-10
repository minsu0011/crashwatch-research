@echo off
setlocal
cd /d "%~dp0"
python -m py_compile surge_precision_gate_v12.py run_surge_precision_gate_v12.py verify_surge_precision_gate_v12.py synthetic_e2e_v12.py test_surge_precision_gate_v12.py
if errorlevel 1 exit /b 1
python -m unittest -v test_surge_precision_gate_v12.py
if errorlevel 1 exit /b 1
if exist synthetic_output_v12 rmdir /s /q synthetic_output_v12
python synthetic_e2e_v12.py
if errorlevel 1 exit /b 1
python verify_surge_precision_gate_v12.py --output synthetic_output_v12
if errorlevel 1 exit /b 1
echo V12 CHECKS PASS
