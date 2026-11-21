# v1 held-out diagnostics (descriptive only)

Source: frozen v1 predictions (`git tag v1-results`, commit 01e43cb), test origins 2024-01-01 to 2025-12-01, n = 593. MAE in percentage points (pp) of capacity. These tables describe v1 and were not used to configure v2.

Season is by forecast origin month. Fullness is storage at origin as % of capacity. For ORO, v1 used the post-2024 capacity (3,424,753 AF) as the denominator for all dates.

## By station_id

| station_id   |   n |   GBM |   Ridge |   Persistence |   30-day trend |   Monthly seasonal |   interval_90_coverage |
|:-------------|----:|------:|--------:|--------------:|---------------:|-------------------:|-----------------------:|
| DNP          | 101 | 1.761 |   2.125 |         3.86  |          2.79  |              2.558 |                  1     |
| EXC          | 101 | 2.311 |   2.976 |         5.596 |          4.334 |              3.212 |                  0.99  |
| FOL          | 101 | 4.533 |   6.173 |         9.806 |          9.242 |              5.47  |                  0.911 |
| NML          | 101 | 1.146 |   1.914 |         1.848 |          1.292 |              1.181 |                  1     |
| ORO          |  88 | 3.833 |   3.434 |         7.21  |          5.318 |              3.234 |                  0.932 |
| SHA          | 101 | 3.433 |   3.837 |         6.599 |          5.886 |              3.303 |                  0.941 |

## By season

| season   |   n |   GBM |   Ridge |   Persistence |   30-day trend |   Monthly seasonal |   interval_90_coverage |
|:---------|----:|------:|--------:|--------------:|---------------:|-------------------:|-----------------------:|
| DJF      | 138 | 4.2   |   4.221 |         5.414 |          5.157 |              4.065 |                  0.92  |
| JJA      | 143 | 1.698 |   3.339 |         8.465 |          5.566 |              3.08  |                  0.993 |
| MAM      | 156 | 3.611 |   4.062 |         6.052 |          5.616 |              3.506 |                  0.942 |
| SON      | 156 | 1.815 |   2.102 |         3.407 |          2.964 |              2.079 |                  0.994 |

## By fullness

| fullness   |   n |   GBM |   Ridge |   Persistence |   30-day trend |   Monthly seasonal |   interval_90_coverage |
|:-----------|----:|------:|--------:|--------------:|---------------:|-------------------:|-----------------------:|
| <40%       |  13 | 8.161 |   8.221 |        10.507 |          9.422 |              8.485 |                  0.692 |
| 40-60%     |  84 | 3.453 |   3.608 |         5.704 |          6.133 |              3.688 |                  0.94  |
| 60-80%     | 252 | 2.445 |   3.205 |         5.582 |          3.456 |              2.688 |                  0.976 |
| >=80%      | 244 | 2.692 |   3.295 |         5.782 |          5.481 |              3.177 |                  0.971 |

## By regime

| regime                |   n |   GBM |   Ridge |   Persistence |   30-day trend |   Monthly seasonal |   interval_90_coverage |
|:----------------------|----:|------:|--------:|--------------:|---------------:|-------------------:|-----------------------:|
| ORO new storage table |  66 | 3.741 |   3.529 |         7.579 |          5.409 |              3.596 |                  0.939 |
| ORO old storage table |  22 | 4.111 |   3.146 |         6.104 |          5.045 |              2.149 |                  0.909 |
| other stations        | 505 | 2.637 |   3.405 |         5.542 |          4.709 |              3.145 |                  0.968 |

## v1 GBM minus monthly seasonal baseline: moving-block bootstrap (2,000 replicates)

| block length | estimate | 95% CI low | 95% CI high | share of replicates GBM better |
|---|---|---|---|---|
| 5 weeks | -0.344 | -0.804 | 0.030 | 0.968 |
| 9 weeks | -0.344 | -0.898 | 0.106 | 0.945 |
| 13 weeks | -0.344 | -0.926 | 0.108 | 0.935 |

## Observations

- See the tables above. Interpretation is in STATUS.md (Checkpoint A).
