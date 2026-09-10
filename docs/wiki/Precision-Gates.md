# 모델 점수와 허용 경보 사이

surge 연구의 목표는 precision 70% 이상이며 최소 경보 수와 경보일·recall·신뢰하한 조건을 함께 본다. 뒤 계열에서는 평가 fold별 최소 30개 alerts가 중요한 기준이다. 버전별 상세 조건은 동일하다고 가정하지 않는다.

좋은 점수 순으로 한두 건만 남겨 precision 100%를 만드는 정책은 충분하지 않다. 경보량과 coverage를 동시에 봐야 한다. threshold를 development에서 한 번 고르고 confirmation으로 넘기는 이유다.

V13 고정 정책의 진단 alert가 있었어도 production은 NO_ALERT였다. 후보 정책의 성능을 측정하는 것과 사용자에게 경보를 내보내는 권한은 다르다.

사후 oracle top-K는 ranking 자체의 한계를 보는 도구다. V13은 이 진단에서도 요구 수준을 만들지 못해 threshold만 조정하는 해결책이 충분하지 않다고 판단했다. oracle을 실행 가능한 정책으로 보고하지 않는다.
