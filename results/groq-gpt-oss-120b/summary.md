Model: `openai:openai/gpt-oss-120b`. 80 instances (8 families, 5 size levels), 3 independent programs per family and condition, up to 3 repair rounds, 30.0 s limit per run. Total: 720 runs.

**Overall**

| Condition | Runs | Valid answer | Correct (valid and optimal) [95% CI] |
|---|---|---|---|
| Direct Python | 240 | 88% | 75% [69-80] |
| CP-SAT (one shot) | 240 | 100% | 99% [97-100] |
| CP-SAT + repair | 240 | 100% | 100% [98-100] |

**By repair round**

| CP-SAT + repair | Correct (cumulative) [95% CI] |
|---|---|
| round 0 (no repair) | 99% [97-100] |
| after 1 repair round(s) | 100% [98-100] |
| after 2 repair round(s) | 100% [98-100] |
| after 3 repair round(s) | 100% [98-100] |

Of 2 programs that were repaired successfully, 2 (100%) also solved a sibling instance of the same family and size that the model never saw.

**By problem family** (correct answers)

| Family | Direct Python | CP-SAT (one shot) | CP-SAT + repair |
|---|---|---|---|
| nqueens_blocked | 100% | 100% | 100% |
| latin_completion | 90% | 100% | 100% |
| graph_coloring | 100% | 100% | 100% |
| knapsack_conflicts | 87% | 100% | 100% |
| bin_packing | 67% | 93% | 100% |
| tsp | 97% | 100% | 100% |
| job_shop | 47% | 100% | 100% |
| shift_scheduling | 13% | 100% | 100% |

**By size level** (correct answers)

| Size level (1 = smallest, 5 = largest) | Direct Python | CP-SAT (one shot) | CP-SAT + repair |
|---|---|---|---|
| size 1 | 88% | 100% | 100% |
| size 2 | 88% | 100% | 100% |
| size 3 | 77% | 100% | 100% |
| size 4 | 77% | 100% | 100% |
| size 5 | 46% | 96% | 100% |

**Outcome of every run**

| Outcome | Direct Python | CP-SAT (one shot) | CP-SAT + repair |
|---|---|---|---|
| ok | 75% | 99% | 100% |
| suboptimal | 13% | 1% | 0% |
| invalid | 4% | 0% | 0% |
| format_error | 0% | 0% | 0% |
| runtime_error | 0% | 0% | 0% |
| timeout | 0% | 0% | 0% |
| no_code | 8% | 0% | 0% |

![correct by size](results/correct_by_size.png)

![correct by family](results/correct_by_family.png)

![repair curve](results/repair_curve.png)
