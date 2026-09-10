# 연구와 예측 인터페이스의 경계

[deployment/crashwatch_predict.py](../../deployment/crashwatch_predict.py)는 별도로 고정한 normal-market 예측 경로다. 여러 surge 연구 중 가장 번호가 큰 폴더를 자동으로 사용하는 구조가 아니다.

```powershell
python deployment/crashwatch_predict.py --help
```

실제 예측에는 해당 모델 가중치와 같은 feature schema·순서·시장 상태의 입력이 필요하다. CLI가 열리는 것은 가중치·데이터가 준비됐다는 뜻이 아니다.

모델의 적용 범위와 원 고지는 [deployment](../../deployment)에서 확인한다. 훈련용 결과 frame, 임의의 synthetic feature 또는 다른 branch checkpoint를 같은 입력으로 대체하지 않는다.
