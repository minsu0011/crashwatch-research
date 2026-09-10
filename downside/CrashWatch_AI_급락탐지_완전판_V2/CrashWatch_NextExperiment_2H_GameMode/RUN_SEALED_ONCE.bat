@echo off
chcp 65001 > nul
echo 이 파일은 LOCKED_PROFILE.json이 LOCKED인 경우에만 사용하십시오.
echo 예: RUN_SEALED_ONCE.bat --selection-output "C:\...\next_experiment_2h_game_v1" --sealed-dataset "C:\...\sealed.parquet" --confirm-sealed-once I_UNDERSTAND_SEALED_IS_ONE_TIME
cd /d "%~dp0"
python run_sealed_once.py %*
pause
