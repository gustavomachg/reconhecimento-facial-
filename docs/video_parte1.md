# Roteiro do vídeo — Parte 1

Tópicos exigidos (§35): problema, datasets, arquitetura, hipóteses, experimentos, treinamento,
resultados, comparação, conclusão e escolha do modelo. Sugestão: 8–12 min.

> **Privacidade no vídeo:** não mostrar fotos de outros participantes do dataset privado, nem nomes
> ou resultados individuais. Mostrar só métricas e gráficos agregados. Para demonstrações, usar
> apenas o próprio aluno ou imagens públicas permitidas.

| # | Tópico | O que mostrar | Tempo |
|---|---|---|---|
| 1 | **Problema** | verificação × identificação; open-set ("pessoa desconhecida"); a pergunta do §6 | 0:45 |
| 2 | **Datasets** | LFW (benchmark, 10 folds), CelebA (fine-tuning, termos), privado (sessões, consentimento, LGPD) | 1:00 |
| 3 | **Arquitetura** | pipeline MTCNN → alinhamento → rede → embedding L2 → cosseno → threshold; camadas do código | 1:15 |
| 4 | **Hipóteses** | tabela de [`docs/experiments.md`](experiments.md): um fator por comparação | 1:00 |
| 5 | **Experimentos** | protocolo: splits sem vazamento, validação × teste, checagens de duplicata e sobreposição | 1:00 |
| 6 | **Treinamento** | parcial × completo (blocos congelados), Triplet (batch-hard) × ArcFace (margem angular); `loss_curves.png` | 1:30 |
| 7 | **Resultados** | tabela do README; `roc_comparison_lfw.png`; histogramas mesma pessoa × diferentes; matriz de confusão | 1:30 |
| 8 | **Comparação** | efeito do fine-tuning, da loss, do backbone (E02 × E08: params e ms/face), comparadores (E06), threshold (E07: FAR × FRR) | 1:30 |
| 9 | **Conclusão e escolha** | modelo vencedor pelo critério do ADR-001, threshold do ADR-003, limitações (dataset pequeno, sem anti-spoofing, viés) | 1:00 |

Comandos úteis para gravar: `python -m facerec.report` (tabela e ROC), abrir `experiments/results/`.
