# V11.1에서 고친 네 가지

1. **lag 탐색의 다중검정.** discovery folds 0~2에서 -5~+5를 탐색한 pair별 max-|corr| 통계를 moving-block permutation으로 평가하고 pair 전체에 BH-FDR을 적용했다.
2. **선택한 lag를 고정.** rolling window마다 좋은 lag를 다시 고르지 않고 discovery의 방향·lag로 안정성을 봤다. residual 계수도 해당 window 이전 자료로 fit한다.
3. **3일 목표에 맞춘 feature.** A→B의 lag가 h이면 B의 origin t에서 미래 k에 대응하는 A 값은 A[t+k-h]다. offset≤0만 쓰므로 lag 4~5에 A[t]를 무조건 넣지 않는다.
4. **같은 행의 BASE/PLUS 비교.** base score가 finite인 동일 source-row 집합에서만 추가 신호를 비교하고, threshold는 development folds 3·4에서 고정해 후속 fold에 적용한다.

보정 후 q≤0.10 관계 27개 중 26개가 lag 0라는 당시 기록이 남았다. 안정적인 일반 directed 관계를 확보하지 못했으므로 인과적 선행 신호라고 주장하지 않았다. lag-0 동조와 선후행은 다른 결과다.

계산량을 줄이기 위한 screening/refinement도 permutation의 작은 p-value 해상도 요구와 함께 설계했다. 단순히 좋은 상관 몇 개를 고르는 것보다 검정 비용과 선택 편향을 함께 다룬 부분이다.
