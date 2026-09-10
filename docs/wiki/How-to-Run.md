# 실험 실행 전에 준비할 것

먼저 [deployment 안내](Deployment-Interface.md)로 예측 인터페이스와 연구 실행이 다름을 확인한다. 연구에는 target-valid feature frame, 고정 fold, 이전 단계 OOF score, A/B evidence 등 branch별 입력이 필요하다.

[surge](../../surge)의 각 `run_*.py`는 해당 실험 전용이다. 일부 wrapper의 기본 입력 경로는 원래 연구 폴더 배치를 가정한다. 경로를 확인하지 않고 전체 batch launcher를 실행하지 말고 실제 parser에 선언된 인자로 입력·출력을 지정한다.

V11.1은 `run_surge_leadlag_v11_1.py`, V13/V14는 각각의 runner와 검증기를 함께 읽는다. synthetic E2E는 계약 검사이며 실제 시장 성과가 아니다. 단순 도움말 실행에 가중치와 원천 데이터를 대신 넣지 않는다.

[데이터 준비](../../data/README.md) · [출처](../../ATTRIBUTION.md)
