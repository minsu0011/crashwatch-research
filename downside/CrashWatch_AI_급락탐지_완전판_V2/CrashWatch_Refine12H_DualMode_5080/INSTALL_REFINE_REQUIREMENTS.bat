@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_refine12h.txt
if errorlevel 1 goto :error

echo Installing PyTorch CUDA 13.0 wheel for RTX 50-series...
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
  echo CUDA 13.0 wheel failed. Trying CUDA 12.8 fallback...
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
)
if errorlevel 1 goto :error

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU not visible')"
python -c "import xgboost, lightgbm, catboost; print('xgboost',xgboost.__version__,'lightgbm',lightgbm.__version__,'catboost',catboost.__version__)"
echo Installation completed.
pause
exit /b 0
:error
echo Installation failed. Check Python 3.11/3.12, NVIDIA driver, and network connection.
pause
exit /b 1
