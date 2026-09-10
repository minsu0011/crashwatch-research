@echo off
setlocal
cd /d "%~dp0"
python -m py_compile surge_precision_gate_v12.py surge_precision_gate_v12_7h.py run_surge_precision_gate_v12.py run_surge_precision_gate_v12_7h.py verify_surge_precision_gate_v12_7h.py synthetic_e2e_v12_7h.py test_surge_precision_gate_v12_7h.py || exit /b 1
python -m unittest -v test_surge_precision_gate_v12_7h.py || exit /b 1
if exist synthetic_output_v12_7h rmdir /s /q synthetic_output_v12_7h
python synthetic_e2e_v12_7h.py || exit /b 1
echo.
echo V12-7H CODE CHECKS PASSED
endlocal
