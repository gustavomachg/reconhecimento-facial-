# Protocolo dos experimentos — Parte 1

Pergunta de pesquisa (enunciado §6):

> **Qual configuração de modelo, treinamento e/ou fine-tuning apresenta o melhor desempenho para o
> problema de reconhecimento facial estudado?**

Cada experimento é **controlado**: muda **um fator** em relação a outro experimento, para que a
diferença de resultado possa ser atribuída a esse fator.

| Comparação | Fator isolado |
|---|---|
| E01 × E02 | arquitetura/loss do pré-treino (FaceNet/VGGFace2 × ArcFace/WebFace600K) |
| E02 × E08 | tamanho do backbone (iResNet-50 × MobileFaceNet), mesma loss e mesmos dados |
| E01 × E03/E04/E05 | efeito do fine-tuning sobre o mesmo modelo de partida |
| E03 × E04 | função de perda (Triplet × ArcFace), mesmos blocos treináveis e mesmos dados |
| E04 × E05 | fine-tuning parcial × completo, mesma loss |
| E06 | comparador (cosseno, euclidiana, protótipo, k-NN, SVM), mesmo embedding |
| E07 | threshold de decisão, mesmo modelo e comparador |

## Regras comuns
- **Seed** 42 em todos (`set_seed`: python, numpy, torch).
- **Splits** (ADR-004): CelebA por identidade (70/15/15); LFW pelos 10 folds oficiais; privado por
  sessão (`s1` galeria, `s2` validação, `s3` teste) com desconhecidos diferentes em val e teste.
- **Seleção** de modelo, hiperparâmetros, comparador e threshold: **só na validação**.
  O teste privado é usado **uma única vez**, no E07.
- **Registro automático** em `experiments/results/<ID>/`: config usada, versões das bibliotecas,
  hardware (CPU/GPU), tempos, métricas e gráficos.
- **Métricas**: Accuracy, Precision, Recall, F1, ROC-AUC, EER, FAR/FRR no threshold, matriz de
  confusão, histogramas "mesma pessoa × diferentes", tempo de inferência e nº de parâmetros.

---

## E01 — Baseline: FaceNet pré-treinada
- **Hipótese:** uma InceptionResnetV1 treinada no VGGFace2 já separa bem as identidades (baseline).
- **Configuração:** [`configs/experiments/E01_facenet_baseline.yaml`](../configs/experiments/E01_facenet_baseline.yaml)
- **Dataset:** LFW (6000 pares, 10 folds); privado — validação.
- **Modelo:** InceptionResnetV1 (VGGFace2), entrada 160×160, recorte MTCNN.
- **Estratégia:** sem treino; comparação por cosseno.
- **Hiperparâmetros:** threshold por fold escolhido nos outros 9 (máx. acurácia).
- **Comando:** `python -m facerec.evaluate --config configs/experiments/E01_facenet_baseline.yaml`
- **Resultado (LFW):** acurácia 99,10% ± 0,43; F1 99,10%; ROC-AUC 0,9991; EER 0,93%;
  FAR 0,97% / FRR 0,83%; 61,7 ms/face em CPU; 23,5 M parâmetros (`experiments/results/E01/`).
- **Interpretação:** baseline forte; os erros se dividem entre falsos aceites (29) e falsos
  rejeites (25). Na 1ª execução (escolhendo o maior rosto) deu 95,90% — ver ablação no README.

## E02 — ArcFace pré-treinado (iResNet-50)
- **Hipótese:** a margem angular aditiva gera embeddings mais compactos por identidade e mais
  separados entre identidades → EER menor que o E01.
- **Configuração:** [`E02_arcface_pretrained.yaml`](../configs/experiments/E02_arcface_pretrained.yaml)
- **Dataset:** LFW; privado — validação.
- **Modelo:** iResNet-50 (insightface `buffalo_l`, WebFace600K), 112×112, alinhamento de 5 pontos.
- **Estratégia:** sem treino; cosseno.
- **Comando:** `python -m facerec.evaluate --config configs/experiments/E02_arcface_pretrained.yaml`
- **Resultado (LFW):** acurácia 99,85% ± 0,20; F1 99,85%; ROC-AUC 0,9995; EER 0,30%;
  FAR 0,00% / FRR 0,30%; 90,3 ms/face; 43,6 M parâmetros.
- **Interpretação:** hipótese confirmada — EER ~3× menor que o E01 e nenhum falso aceite em 3000
  pares de pessoas diferentes. Parte do ganho também vem do pré-treino (WebFace600K) e do
  alinhamento por 5 pontos, não só da loss.

## E03 — FaceNet, fine-tuning parcial + Triplet Loss
- **Hipótese:** treinar só os últimos blocos adapta a rede ao domínio sem overfitting.
- **Configuração:** [`E03_facenet_partial_triplet.yaml`](../configs/experiments/E03_facenet_partial_triplet.yaml)
- **Dataset:** treino/validação = CelebA (identidades disjuntas, sem as suspeitas de estar no LFW);
  avaliação = LFW + validação privada.
- **Modelo:** InceptionResnetV1; treináveis: `repeat_3`, `block8`, `last_linear`, `last_bn`.
- **Estratégia:** fine-tuning parcial; Triplet Loss com batch-hard mining (margem 0,2), batches P×K = 16×4.
- **Hiperparâmetros:** Adam, LR 1e-4, weight decay 5e-4, cosine annealing, 15 épocas,
  early stopping (paciência 4) pelo EER de validação; augmentation: recorte, flip, cor, rotação ±10°, cinza.
- **Comandos:** `python -m facerec.train --config …E03…` e `python -m facerec.evaluate --config …E03…`
- **Resultado:** _a preencher (`train_summary.json`, `loss_curves.png`, `metrics.json`)_
- **Interpretação:** _a preencher (ver `best_epoch`: 0 = o fine-tuning não superou o pré-treinado)_

## E04 — FaceNet, fine-tuning parcial + ArcFace Loss
- **Hipótese:** com os mesmos blocos do E03, a ArcFace converge melhor com poucos dados, porque usa
  todas as classes em cada passo e não depende de minerar triplets.
- **Configuração:** [`E04_facenet_partial_arcface.yaml`](../configs/experiments/E04_facenet_partial_arcface.yaml)
- **Diferença em relação ao E03:** só a loss (ArcFace, s = 30, m = 0,5; batch 64).
- **Resultado:** _a preencher_
- **Interpretação:** _a preencher_

## E05 — FaceNet, fine-tuning completo
- **Hipótese:** treinar a rede toda dá ganho pequeno ou overfitting em relação ao parcial, com custo
  de treino bem maior.
- **Configuração:** [`E05_facenet_full_finetune.yaml`](../configs/experiments/E05_facenet_full_finetune.yaml)
- **Diferença em relação ao E04:** todos os blocos treináveis; LR 10× menor (1e-5) para não
  destruir os pesos pré-treinados.
- **Resultado:** _a preencher (comparar `n_trainable_parameters` e `train_time_s` com o E04)_
- **Interpretação:** _a preencher_

## E06 — Comparadores (open-set, validação privada)
- **Hipótese:** com embeddings L2-normalizados, cosseno e euclidiana empatam (‖a−b‖² = 2 − 2·cos);
  protótipos e k-NN são robustos com várias fotos por pessoa; o SVM sofre com poucas amostras por
  classe e não modela "desconhecido".
- **Configuração:** [`E06_matchers_open_set.yaml`](../configs/experiments/E06_matchers_open_set.yaml)
  — trocar `model` pelo vencedor de E01–E05/E08.
- **Métricas:** acurácia open-set, F1 macro, FAR de desconhecidos, FRR de conhecidos,
  taxa de confusão entre cadastrados, ms por consulta.
- **Resultado:** _a preencher_
- **Interpretação:** _a preencher_

## E07 — Threshold (validação → teste único)
- **Hipótese:** existe um threshold que equilibra FAR e FRR; escolhido na validação, ele se mantém no teste.
- **Configuração:** [`E07_threshold_sweep.yaml`](../configs/experiments/E07_threshold_sweep.yaml)
- **Estratégia:** varredura de 101 thresholds na validação; escolha por EER (ou FAR-alvo de 1%);
  avaliação única no teste com o valor congelado.
- **Resultado:** _a preencher (`threshold_sweep.png`, `test_score_hist.png`, `metrics.json`)_
- **Interpretação:** _a preencher_

## E08 — ArcFace com backbone MobileFaceNet (custo × desempenho)
- **Hipótese:** um backbone leve treinado com a mesma loss e os mesmos dados do E02 perde pouco
  desempenho e é muito mais rápido.
- **Configuração:** [`E08_mobilefacenet_pretrained.yaml`](../configs/experiments/E08_mobilefacenet_pretrained.yaml)
- **Modelo:** MobileFaceNet (insightface `buffalo_s`, `w600k_mbf.onnx`), 112×112 — usa as mesmas faces
  alinhadas do E02 (`aligned_arcface`).
- **Comando:** `python scripts/download_arcface.py --pack buffalo_s` e
  `python -m facerec.evaluate --config configs/experiments/E08_mobilefacenet_pretrained.yaml`
- **Resultado (LFW):** acurácia 99,67% ± 0,20; F1 99,67%; ROC-AUC 0,9993; EER 0,32%;
  FAR 0,27% / FRR 0,40%; **5,0 ms/face**; **3,4 M parâmetros**.
- **Interpretação:** hipótese confirmada — perde 0,18 ponto de acurácia em relação ao E02 (dentro
  do desvio entre folds), sendo ~18× mais rápido e ~13× menor. Melhor custo × desempenho.
