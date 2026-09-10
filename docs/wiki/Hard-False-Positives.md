# 높은 점수의 오답을 별도로 보기

V7은 A=상위 점수 TP, B=상위 점수 FP, C=낮은 점수 양성으로 오류 집단을 나눴다. A와 B가 어떻게 다른지, C를 B보다 위에 올릴 정보가 있는지가 핵심 질문이다.

pairwise ranker는 C와 B의 순서를 직접 학습하려는 시도다. 최근 126~756거래일 specialist는 과거 전체와 최근 상태의 차이를 보고, 연속 양성 run을 event로 묶는 방식은 같은 사건의 여러 행이 학습을 지배하는 문제를 줄인다. 날짜·시장·bucket 상대 feature와 forward-only meta도 비교했다.

당시 fold 기록에서는 선택·확인·recent 전 구간에 허용 정책이 없어 no-alert가 남았다. 이는 경보 precision이 0이라는 뜻이 아니라 정의할 경보가 없는 상태다. ROC-AUC가 0.5를 넘는 경우에도 최소 수량·시간 전이·precision 조건을 충족하지 못할 수 있다.

V14의 hard-FP stack은 이 문제를 다시 다루되 expert 점수의 cross-fit 경계를 사용한다. 모델이 학습한 행의 정답을 이미 아는 점수를 meta 학습에 바로 넣지 않도록 구분하는 것이 중요하다.
