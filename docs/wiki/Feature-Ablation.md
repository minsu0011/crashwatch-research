# feature가 무엇을 배웠는지 확인하기

feature 수를 늘리면 동일한 정보의 변형도 늘어난다. V13에서는 439개 raw feature와 ticker·횡단면 변환을 포함한 3,511개 후보가 있었다. 작은 large-move 표본에서 상관된 변형을 넓게 탐색하면 우연한 상위 feature가 생길 수 있다.

그래서 block·component ablation을 통해 A/B evidence, lag-0 network, self-state의 기여를 분리했다. 후속 V14에서는 coverage·worst-fold 조건과 상관 중복 제거, discovery 내부의 다른 fold에서 정한 feature 방향을 held-out discovery fold에 적용하는 방식을 넣었다.

V13 primary의 direction 입력에서는 volatility·range·tail·spread 계열을 제한했다. 방향을 묻는 head가 쉬운 “큰 움직임” 문제를 다시 학습하지 않게 하려는 선택이다. signed skew처럼 방향성이 있는 비대칭 정보는 별도로 다룬다.

ablation은 원인 후보를 좁히는 진단이다. 임의의 한 제거 실험을 확인한 뒤 최종 평가 구간까지 이용해 모든 구성 요소를 다시 선택하지 않는다.
