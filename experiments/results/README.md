# Resultados (somente agregados)

Cada experimento escreve em `experiments/results/<ID>/`:

| Arquivo | Conteúdo |
|---|---|
| `config_used.yaml` | cópia exata da config usada |
| `metrics.json` | métricas agregadas (LFW 10-fold, validação privada, custo, ambiente/hardware) |
| `train_summary.json` | (E03–E05) histórico por época, nº de parâmetros treináveis, tempo |
| `split.json` | (E03–E05) identidades CelebA de cada conjunto (IDs públicos numéricos) |
| `loss_curves.png` | curvas de loss treino/validação e EER por época |
| `hist_*.png`, `roc_*.png` | histogramas "mesma pessoa × diferentes" e curvas ROC |
| `matchers_comparison.png` | (E06) comparação dos comparadores |
| `threshold_sweep.png`, `test_score_hist.png` | (E07) varredura do threshold e scores no teste |

`python -m facerec.report` gera `results_table.md` e `roc_comparison_*.png`.

Nada aqui pode conter nomes, IDs de participantes, imagens de rostos ou embeddings.
