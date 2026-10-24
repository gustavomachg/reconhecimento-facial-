# Notebooks de análise

Use notebooks apenas para analisar os arquivos **agregados** de `experiments/results/`
(`metrics.json`, `train_summary.json`). Regras:

- nunca exibir imagens do dataset privado;
- limpar os outputs antes de versionar (`jupyter nbconvert --clear-output --inplace *.ipynb`);
- nada de métricas por participante.

Para a tabela do README e a ROC comparativa, prefira `python -m facerec.report`.
