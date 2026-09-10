# 종목 차이를 신호로 착각하지 않기

전체 표본의 상관이나 feature 중요도가 특정 종목의 반복 사건에 기대고 있을 수 있다. downside의 ticker-independent 계열과 surge의 tickerwise map은 이 문제를 다른 방향에서 본다.

ticker 내부의 과거 z-score·rank·change와 날짜별 시장·bucket 상대 순위는 서로 다른 비교 기준이다. 과거에 알려지지 않은 전체 ticker 이력 percentile이나 신뢰할 수 없는 industry metadata를 primary feature에 쓰지 않는다.

V12는 A/B spec이 있는 종목만 남기면서 실제 stress 범위가 6종목으로 좁아졌다. V13은 전체 target-valid frame을 유지하고 근거가 없는 종목의 evidence만 0으로 표시했다. “근거가 없다”와 “분석 대상이 아니다”는 다른 상태다.

이 복원은 48종목 내 scope 문제를 고친 것이다. 새로운 종목과 다른 시장에서도 일반화됐다는 검증을 뜻하지 않는다.
