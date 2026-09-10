# 레거시 참고자료 해석 규칙

## 그대로 재사용 가능한 것

- 피처 이름과 품질 감사
- 피처-피처 Pearson/Spearman/within-date/within-ticker 상관구조
- 중복 피처 cluster와 대표 피처 후보
- 종목-업종 매핑
- walk-forward 구현 방식, 캐시·원자 저장·단일 클래스 처리 방식

## 구조만 참고하고 급등 target으로 재계산할 것

- `feature_target_correlation_by_fold.csv`
- 개별·조건부·cluster 이탈 효과와 `feature_master_decision.csv`
- 기존 feature importance
- 약종목·약업종 판정
- 정상장 게이트와 regime별 우열
- 기존 모델 후보의 순위

급락 예측에 유용한 피처가 급등 예측에는 무용하거나 반대 방향일 수 있다. 기존 이탈 결과를 그대로 피처 삭제 근거로 쓰지 않는다. 첫 급등 baseline은 가능한 넓은 유효 피처 집합과 P2/P7 profile을 모두 비교하고, 이후 급등 전용 이탈 결과로 축소한다.
