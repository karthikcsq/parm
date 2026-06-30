# Benchmark Results

Generated on the 50-case synthetic V1 dataset.

| Baseline | Correct surfaced suggestion rate | Precision | Recall | Useless intervention rate |
| --- | ---: | ---: | ---: | ---: |
| `no_memory` | 0.00 | 0.00 | 0.00 | 0.00 |
| `input_rag` | 0.00 | 0.00 | 0.00 | 0.00 |
| `prompted_memory_tool` | 0.00 | 0.00 | 0.00 | 0.00 |
| `parm_oracle_monitor` | 1.00 | 1.00 | 1.00 | 0.00 |

The oracle monitor is intentionally gold-assisted in V1. Its role is to prove
that the dataset contains recoverable output/tool-conditioned memory
interventions before later work adds noisy cue selection, hybrid memory search,
reranking, FLARE-style retrieval, or proactive-agent baselines.

The canonical 50-case dataset now covers 10 domains with 5 cases each:
introductions, customer discovery, opportunity/risk, travel planning, health
admin, learning/research, event planning, hiring, personal finance, and home
operations.
