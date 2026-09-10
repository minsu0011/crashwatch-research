# 각 변경이 추가한 질문

| 계열 | 맡긴 역할 | 채택·보류 판단 |
| --- | --- | --- |
| 기본·dual 모델 | 이벤트 점수의 비교 기준 | 급락/급등 평가 계약 분리 |
| V3/V4 | feature family와 학습기 비교 | 모델 이름 증가와 경보 개선 구분 |
| V5/V6 | alert budget·precision 정책 | 작은 경보 수로 만든 높은 적중률 제한 |
| V7 | hard-FP pairwise·recent specialist | gate 미달, 오류 지도 유지 |
| V8–V10 | organic·tickerwise separation | 시점 안전하지 않은 변환 제외 |
| V11/11.1 | 독립 선후행 축 | 일반 directed edge를 primary에 미채택 |
| V12/13 | precision stress → 크기/방향 분리 | 전체 표본 복원, 방향 분리 병목 확인 |
| V14 | direct/hazard/downside/cross-fit | 신규 가설, 검증 완료 모델로 미승격 |

V13에서 V11 lag-0 정보는 제한적으로 남겼다. self-state는 제거한 ablation이 더 나은 기록을 보여 V14 primary에서 제외했고 directed edge는 진단용으로만 보존했다. 다만 ablation의 세부 개선치를 다른 fold·표본의 값과 빼서 보고하지 않는다.

[실험 결정](Experiments-and-Decisions.md) · [결과](Validation-and-Results.md)
