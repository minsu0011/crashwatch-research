# GPU 실행 조건

- XGBoost는 `tree_method="hist"`, `device="cuda"` 경로를 쓴다.
- CatBoost는 `task_type="GPU"`, `devices="0"`을 쓴다. GPU reduction 때문에 반복 실행이 완전히 결정적이지 않을 수 있다.
- Windows LightGBM은 GPU/OpenCL 경로를 사용하며 사용할 수 없으면 CPU로 전환하고 기록한다.
- RTX 50 계열의 PyTorch 실행에는 Blackwell을 지원하는 wheel이 필요하다. 설치 스크립트는 CUDA 13.0 인덱스를 먼저 시도하고 CUDA 12.8을 대안으로 둔다. 실제 드라이버·wheel 호환성을 확인해야 한다.
- 전체 실험 전 `CHECK_GPU_STACK.bat`로 작은 학습을 확인한다. import 성공만으로 GPU 학습 경로가 작동한다고 보지 않는다.
