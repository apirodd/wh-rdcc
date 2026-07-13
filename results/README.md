# Results

JSON files with the metrics reproducing the paper tables. Each is produced by
the corresponding script in `scripts/` and can be regenerated from scratch.

| File | Produced by | Paper table |
|---|---|---|
| `phase2_results.json` | `scripts/phase2_statistics.py` | Multi-dataset validation (mean ± std) |
| `phase3_ablation.json` | `scripts/phase3_ablation.py` | Ablation on d, L, Δ |
| `phase4_baselines.json` | `scripts/phase4_baselines.py` | Comparison with LSTM-CC / JEPA-CC |
| `phase5_overhead.json` | `scripts/phase5_overhead.py` | Computational overhead |

Each metrics entry contains: `mae`, `tw`, `ct`, `pe`, `sr`, and (phases 3–4)
`erank` (effective rank of the latents).
