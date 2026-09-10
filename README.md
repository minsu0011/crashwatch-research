# CrashWatch Research

큰 가격 변동을 감지하는 것과 실제로 맞는 경보를 내는 것은 다르다. CrashWatch는 급락 탐지에서 출발해 급등 이벤트, 종목 간 일반화, 반복되는 오경보를 따로 연구한 프로젝트다. 변동성 신호가 강해도 방향을 잘못 맞히면 경보로 쓰기 어렵다는 점 때문에, 모델 점수뿐 아니라 경보 수·precision·시간 전이 조건까지 함께 설계했다.

## 구조와 사용 기술

가격·거래량·시장·재무 feature → 시점별 이벤트 label → forward split → 기본 분류·ablation → 종목별 상대 신호·hard-FP·lead-lag 연구 → 고정 경보 정책

- [downside](downside): 급락·dual-mode 및 종목 일반화 연구
- [surge](surge): 급등 방향, feature, 경보 정책의 실험 계보
- [deployment](deployment): 별도로 고정한 normal-market 예측 인터페이스

Python, pandas·NumPy, scikit-learn, LightGBM, XGBoost, CatBoost와 Parquet을 사용한다. 모든 실험을 하나의 최신 모델로 연결하지 않고 목적과 평가 경계를 구분했다.

## 데이터와 목표

주요 surge 목표는 거래일 t에서 D+1~D+3 중 누적 종가수익률이 한 번이라도 +5%에 도달하는 사건이다. 미래 경로는 label을 만드는 데만 사용한다. 급락 branch와 surge branch의 목표·유니버스가 같다고 가정하지 않는다.

V13 연구 기록의 범위는 48종목, target-valid 91,775행, 원본 feature 439개다. discovery, development validation, confirmation, recent diagnostic을 나누고 feature·threshold를 고른 뒤 후속 구간에 고정 적용한다.

## 개발 과정

1. **급락·dual-mode 탐지에서 출발했다.** 시장 상태와 종목 특성이 섞이는 문제를 분리하기 위해 feature ablation과 ticker-independent 평가를 확장했다.
2. **좋은 분류 점수가 곧 좋은 경보는 아니었다.** model zoo 이후 경보 예산과 precision 조건을 따로 평가했다. 몇 건만 골라 높은 적중률을 만드는 정책을 막기 위해 최소 경보·경보일·recall도 함께 봤다.
3. **상위 오경보를 별도 문제로 정의했다.** V7에서 높은 점수의 진짜 양성·거짓 양성, 낮은 점수의 양성을 나눠 오류 지도를 만들었다. pairwise ranker와 recent specialist를 추가했지만 70% precision gate를 만족하는 정책을 확보하지 못했다.
4. **종목별 상대 신호와 선후행을 조사했다.** 전체 상관으로는 특정 종목·시장 상태의 효과가 섞일 수 있어 ticker transform, bucket residual과 lead-lag를 분리했다.
5. **V11.1에서는 선후행 검증 자체를 고쳤다.** lag 탐색의 다중검정, rolling window마다 lag를 다시 고르는 문제, 3일 목표와 leader feature의 정렬을 교정했다. 일반적인 directed edge는 주력 입력으로 채택하지 않았다.
6. **V12의 좁아진 평가 범위를 V13에서 복원했다.** A/B 근거가 있는 6종목만 남았던 문제를 풀어 48종목 전체를 유지했다. 근거가 없는 행은 삭제하지 않고 evidence를 0으로 표시했다.
7. **큰 움직임과 상승 방향을 분리했다.** V13은 `P(move) × P(up | move)`를 시도했으나 방향 분리가 약했다. 사후 top-K 진단으로도 경보 기준에 이르지 못해 threshold만 바꾸는 접근을 중단했다.
8. **V14는 서로 다른 실패 원인을 보는 expert를 병렬로 뒀다.** direct surge, up-crossing hazard, downside competing risk, 기존 magnitude/direction expert와 hard-FP cross-fit stack을 연구한다. 이를 V13을 대체한 검증 완료 모델로 선언하지 않는다.

## 모델과 정책의 역할

| 구성 | 역할 |
| --- | --- |
| 기본 분류기 | 원본·변환 feature에서 이벤트 점수 생성 |
| ablation·상대 feature | 어떤 정보가 특정 종목이나 크기 효과에 기대는지 진단 |
| hard-FP ranker·specialist | 높은 점수의 오경보와 놓친 양성의 순서 구분 |
| lead-lag | 선행 정보가 추가 신호가 되는지 독립적으로 검증 |
| magnitude/direction | 큰 변동과 상승 여부라는 두 질문을 분리 |
| competing-risk·cross-fit stack | 상승 경로와 하락 위험, expert 간 중복을 함께 반영 |
| precision gate | 점수를 실제 경보로 내보낼지 결정; 조건 미달이면 경보 없음 |

## 결과와 한계

당시 실험 기록 기준 V7은 precision gate를 통과하지 못했고, V13의 고정 정책도 confirmation fold 5·6에서 precision 37.11%·33.33%로 목표에 미달했다. 정책 후보의 진단 경보와 실제 배포 허용 경보는 다르다.

V11.1의 보정 뒤 유의 관계 대부분은 lag 0였고 안정적인 일반 선형 directed lead-lag를 확보하지 못했다. 동조 관계가 있다는 이유로 인과적인 선행 신호라고 부르지 않았다.

branch별 평가 조건이 달라 직접적인 개선치로 비교하지 않았다. [별도 deployment 인터페이스](deployment)는 최신 연구 branch의 성과를 자동 상속하지 않는다. 모델 파일과 입력 schema, 적용 시장 상태가 함께 맞아야 한다.

## 시작하기

```powershell
python deployment/crashwatch_predict.py --help
```

이 명령은 예측 인터페이스 안내다. 학습용 feature frame, fold 및 학습 가중치는 별도 준비해야 하며, 실행별 인자는 [실행 안내](docs/wiki/How-to-Run.md)에서 구분했다. 외부 구현과 데이터 이용 범위는 [출처](ATTRIBUTION.md)에 남겼다.

## 상세 문서

[연구 지도](docs/wiki/Home.md) · [개발 과정](docs/wiki/Development-Journey.md) · [모델 발전](docs/wiki/Model-Evolution.md) · [오경보](docs/wiki/Hard-False-Positives.md) · [선후행 교정](docs/wiki/Lead-Lag.md) · [실험 결정](docs/wiki/Experiments-and-Decisions.md) · [결과](docs/wiki/Validation-and-Results.md)
