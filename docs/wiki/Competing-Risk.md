# V14의 병렬 expert

Direct surge head는 공식 3일 상승 target을 바로 학습한다. Up-crossing hazard는 D1/D2/D3 중 처음 +5%에 닿는 날을 보조 target으로 만들어 1-(1-h1)(1-h2)(1-h3) 형태로 결합한다.

Downside head는 같은 미래창의 -5% 도달 위험을 추정한다. 높은 변동성이 상승과 하락을 함께 암시하는 경우를 구분하기 위한 veto 정보이며 stack에는 1-P(downside)가 들어간다.

기존 V13 magnitude/direction은 또 하나의 expert다. 이를 유일한 최종 score로 강제하지 않는다. hard-FP 정보와 cross-fit expert score를 이용한 결합은 별도 학습 단계다.

미래 first-hit-day와 경로 수익률은 보조 label을 만드는 용도다. feature로 전달해서는 안 된다. 상·하단 모두 도달하는 경로의 의미도 확인해야 한다.

이는 구현된 연구 가설의 설명이다. V14라는 번호나 실행 코드의 존재만으로 precision gate 통과·배포 승격을 주장하지 않는다.
