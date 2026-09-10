# 경보가 실패한 이유를 좁혀 간 과정

급락·dual-mode 연구에서 시작해 feature ablation과 ticker-independent 평가로 확장했다. 이후 surge의 분류 성능과 실제 경보 품질을 분리하면서 V3 all-feature ablation, V4 model zoo, V5 alert budget, V6 precision gate 계열이 이어졌다.

V7은 높은 점수의 false positive가 단순 threshold 조정으로 없어지지 않는 문제를 직접 다뤘다. 상위 TP, 상위 FP, 낮은 점수의 양성을 나누고 pairwise·recent·event-balanced specialist를 만들었다. 하지만 gate를 만족하는 정책이 없어 새로운 미래 holdout을 소비하는 대신 오류 지도를 남겼다.

V8 organic ablation, V9 separation map, V10 tickerwise 계열에서는 신호가 어느 종목·상태에 있는지를 더 세밀하게 봤다. 전체 기간 percentile을 이용하는 일부 변환과 잘못된 industry metadata는 후속 primary에서 제외했다.

V11의 선후행 연구는 독립 가지였다. V11.1에서는 lag 탐색 다중검정과 rolling lag 재최적화, target과 feature의 시점 정렬 문제를 교정했다. 교정 결과를 확인한 뒤 directed edge를 주력 모델에 다시 넣지 않았다.

V12는 hard-FP precision stress를 시도했지만 A/B spec이 있는 6개 종목만 남는 범위 문제가 있었다. V13에서 전체 48종목·91,775행을 복원하고 크기/방향 모델을 분리했다.

V13에서도 방향 ranking이 약하자 V14는 단일 확률곱을 유일한 score로 쓰지 않고 expert 중 하나로 낮췄다. direct target과 first-hit hazard, downside veto와 hard-FP stack을 나눠 비교하는 것이 다음 가설이다. 가장 최신 번호가 검증 완료를 뜻하지 않는다.
