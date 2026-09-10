# 데이터·모델·경보를 분리하기

[downside](../../downside), [surge](../../surge), [deployment](../../deployment)는 세 가지 다른 목적의 경로다. 급락 연구의 평가 조건을 surge의 +5% target과 섞지 않고, 최신 연구 branch를 deployment의 모델이라고 부르지 않는다.

surge에서는 t까지의 가격·시장·재무 feature를 만들고 D+1~D+3 미래 경로로 label을 생성한다. forward fold를 사용해 discovery·선택·confirmation을 나눈다. 모델은 score를 내고, 정책은 최소 경보 수와 precision 조건에 따라 score를 경보로 전환한다.

V14의 expert는 direct surge, up-crossing hazard, downside risk와 V13의 magnitude/direction이다. meta 입력은 가능한 한 cross-fit으로 만들고 hard-FP를 별도로 구분한다. 이러한 분해는 단순한 모델 평균이 아니라 서로 다른 오류를 보는 역할 분리다.

현재 저장소에는 원본 feature frame·가중치 전체가 포함되지 않는다. 입력과 checkpoint의 feature 순서, 학습 범위, 모델 버전이 맞아야 실제 예측 결과를 읽을 수 있다.
