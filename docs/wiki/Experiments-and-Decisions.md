# 유지하거나 제외한 근거

V7에서는 오류 지도를 다음 연구의 입력으로 남기되 통과 정책이 없는 상태에서 새 holdout을 열어 threshold를 찾지 않았다. V11.1에서는 directed lead-lag를 다시 보정했고 유의한 동조를 선행 인과로 승격하지 않았다.

V13은 6종목 probe를 전체 연구처럼 해석하는 scope 문제를 고쳤다. industry rank와 전체 ticker 이력 percentile 계열은 primary에서 제외했고, 방향 head가 크기 정보에 의존하는 문제를 분리했다.

V14에서는 V13의 A/B expert를 유지하지만 self-state와 실패한 directed edge를 primary에서 제외한다. sparse feature의 discovery 선택은 fold-safe 방향·coverage·worst-fold·중복 제거로 제한한다.

이 결정들은 점수를 높이는 것뿐 아니라 무엇을 같은 실험이라고 부를 수 있는지 정하는 기준이다. 대상 행·label·선택 경계가 바뀐 branch 간에는 직접적인 개선치를 보고하지 않는다.
