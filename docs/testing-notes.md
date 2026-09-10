# 테스트와 실행 범위

저장소 루트에서 의존성을 준비한 뒤 다음 범위를 확인할 수 있다.

```powershell
python deployment/crashwatch_predict.py --help
```

확인한 것은 고정 예측 CLI의 도움말이다. branch별 전체 학습, 70% precision 달성, deployment 가중치를 포함한 실데이터 예측의 재현을 뜻하지 않는다.

급락·급등 branch의 목표와 평가 구간은 다르다. V7·V13의 경보 정책은 요구 precision을 충족하지 못했다. 배포 인터페이스는 별도 가중치가 필요하며 최신 실험의 성과를 자동으로 이어받지 않는다.

테스트 실행과 전체 원천 수집·학습은 별개다. 연구 결과는 [README](../README.md)의 당시 기록과 입력 조건을 함께 읽는다.
