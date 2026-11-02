# Cross-corpus teaching material (ILLUSTRATIVE)

**ILLUSTRATIVE — not measured from a real run**

Replicates **only** the paper §4.9 / Table 14 protocol:

- Train on **PlantVillage**, test on **PlantDoc** (no target fine-tuning)
- Train on **PlantDoc**, test on **PlantVillage** (reverse)

Four bars per direction = architecture variants (`baseline` → `multiscale` → `attention` → `multiscale_attention`), same idea as the ablation table.

**Not included:** `finetune` / `scratch` (those are repo extensions, not in the paper).

| Reference | PV→PD | PD→PV |
|-----------|-------|-------|
| Paper Table 14 | 41.00 % | 53.23 % |
| Illustrative variants | baseline: 55.2%, multiscale: 56.2%, attention: 57.6%, multiscale_attention: 59.4% | baseline: 54.8%, multiscale: 55.8%, attention: 57.1%, multiscale_attention: 58.9% |

Main figure: `cross_corpus_by_variant.png`. Summary: `cross_corpus_summary.csv`.

```bash
python generate_cross_corpus_teaching_figures.py
```
