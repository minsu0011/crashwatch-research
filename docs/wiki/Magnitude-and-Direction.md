# 변동의 크기를 맞히면 방향도 맞힐까

V12의 direction gate에서 volatility·range·tail feature가 많은 비중을 차지했다. 모델이 상승 방향 대신 큰 움직임 자체를 반복해서 학습한다는 문제가 있었다.

V13은 P(절대 ±5% move)와 P(+5% surge | move)를 나눠 곱했다. large-move가 없는 행과 큰 움직임 중 방향이 틀린 행의 실패를 따로 보기 위한 설계다. 최종 surge target은 그대로 유지했다.

당시 전체 target-valid 자료에서 large-move 양성은 25,829건, surge 양성은 14,314건이었다. 큰 변동이 있는 사례에서도 상승은 당연한 결과가 아니다. direction head와 희소한 feature 탐색의 불안정성이 후속 병목으로 남았다.

V14에서는 이 확률곱을 삭제하지 않고 하나의 expert로 남겼다. direct surge가 놓치는 상호작용을 보완하는지, downside veto가 오경보를 줄이는지 함께 평가하는 쪽으로 질문을 바꿨다.
