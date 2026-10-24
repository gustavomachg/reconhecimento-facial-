# Parte 1 — rastreabilidade enunciado → repositório

| § do enunciado | Exigência | Onde está |
|---|---|---|
| 6 | investigação experimental, não só usar um modelo pronto | 8 experimentos controlados — [`docs/experiments.md`](experiments.md) |
| 7 | pipeline detecção → alinhamento → rede → embedding → comparação → reconhecimento | `detection.py` → `models/` → `matching.py` → `threshold.py` |
| 7 | escolha de modelo justificada (FaceNet, ArcFace, MobileFaceNet…) | [ADR-001](decisions/ADR-001-face-model.md); E01, E02, E08 |
| 8–9 | dataset público + privado | privado: pasta `pessoas/` avaliada por `python -m facerec.evaluate_folder`; público: LFW/CelebA — README e [`data/README.md`](../data/README.md) |
| 10 | fonte, licença, finalidade, uso; sem redistribuição | tabela "Datasets" do README; `scripts/download_lfw.py`, `scripts/download_celeba.md` |
| 11 | sem scraping | nada de scraping no código; [`docs/privacy.md`](privacy.md) |
| 12 | dataset privado com variações e minimização | [`data/README.md`](../data/README.md) (sessões `s1/s2/s3`, nº mínimo de fotos) |
| 13–16 | LGPD, consentimento, termos fora do GitHub, adultos | [`docs/privacy.md`](privacy.md); `consent/` ignorado e bloqueado |
| 17–20 | fotos, embeddings, bancos, checkpoints e logs privados não vão ao GitHub | `.gitignore` (qualquer coisa em `data/` exceto o README, `database/`, `embeddings/`, `checkpoints/`, `*.pt`, `*.onnx`…), hook `scripts/check_private_files.py` |
| 18 | conferir o histórico do Git | `python scripts/check_private_files.py --history` |
| 21 | o que pode ir para o GitHub | código, specs, testes, configs, docs, métricas e gráficos agregados |
| 22 | dados no computador, fora de serviços de compartilhamento | ver alerta sobre OneDrive em [`docs/privacy.md`](privacy.md) |
| 23 | agentes de IA não recebem fotos reais | testes só com dados sintéticos; `scripts/self_test_photos.py` roda localmente e imprime só números |
| 24 | eliminação ao fim e exclusão por pessoa | `scripts/purge_private_data.py`; remover `data/private/raw/<ID>` |
| 27 | hipótese, config, dataset, modelo, estratégia, hiperparâmetros, métricas, resultado, interpretação | [`docs/experiments.md`](experiments.md) + YAMLs + `experiments/results/<ID>/` |
| 28 | pré-treinado × parcial × completo; comparadores; losses | E01/E02/E08 × E03/E04 × E05; E06; Triplet × ArcFace (E03 × E04) |
| 29 | treino / validação / teste; teste nunca usado para escolhas | [ADR-004](decisions/ADR-004-data-split-and-leakage.md); teste privado só no E07 |
| 30 | data leakage (mesma imagem, quase-duplicatas, augmentation, influência do teste) | `split_by_identity`, `split_private_by_session`, `scripts/check_duplicates.py`, `scripts/check_identity_overlap.py`, augmentation só no treino |
| 31 | Accuracy, Precision, Recall, F1, ROC-AUC, FAR, FRR, EER, matriz de confusão, distribuições | `metrics.py`, `evaluate.py` (`hist_*.png`, `roc_*.png`, `confusion_*.png`) |
| 32 | threshold reconhecido/desconhecido justificado experimentalmente | E07, [ADR-003](decisions/ADR-003-recognition-threshold.md) |
| 33 | análise (melhor modelo, por quê, fine-tuning, threshold, erros, limitações, custo) | seção "Análise" do README (a preencher com os resultados) |
| 34 | seed, versões, hiperparâmetros, arquitetura, épocas, optimizer, LR, batch, transforms, hardware | YAMLs + `environment` em `metrics.json`/`train_summary.json` |
| 35 | README com tabela, instruções de reprodução, vídeo | README; `python -m facerec.report`; [`docs/video_parte1.md`](video_parte1.md) |
