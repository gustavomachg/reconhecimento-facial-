# ADR-001 — Modelo de reconhecimento facial

**Status:** aceito (MobileFaceNet, E08) — revisável após o E06/E07 no dataset do grupo
**Data:** 2026-10-24

## Contexto
Precisamos de uma rede que gere embeddings faciais discriminativos, rode em notebook
(RTX 4050 Laptop / CPU) e possa ser usada numa aplicação local.

## Candidatos
| Modelo | Arquitetura / treino | Entrada | Parâmetros | Licença dos pesos |
|---|---|---|---|---|
| FaceNet (`facenet-pytorch`) | InceptionResnetV1, VGGFace2 | 160×160, recorte MTCNN | ~23,5 M | MIT (código); pesos derivados do VGGFace2 (uso de pesquisa) |
| ArcFace (`insightface` buffalo_l `w600k_r50`) | iResNet-50, ArcFace loss, WebFace600K | 112×112, alinhamento 5 pontos | ~43,6 M | **somente não comercial / acadêmico** |
| MobileFaceNet (`insightface` buffalo_s `w600k_mbf`) | MobileFaceNet, ArcFace loss, WebFace600K | 112×112, alinhamento 5 pontos | ~3,4 M | **somente não comercial / acadêmico** |

O MobileFaceNet (E08) permite medir a relação custo × desempenho: mesma loss e mesmos dados de
pré-treino do E02, com um backbone cerca de 13× menor.

Os modelos ArcFace são executados via **onnxruntime** a partir dos ONNX oficiais (`buffalo_l`/`buffalo_s`),
sem instalar o pacote `insightface` (que exige compilação C++ no Windows). O pré-processamento
(alinhamento por similaridade para o template de 5 pontos, normalização `(x-127.5)/127.5`, RGB)
reproduz o do insightface.

Somente a FaceNet é fine-tunada (E03–E05): temos os pesos em PyTorch e o código de treino
aberto; os modelos ArcFace são usados congelados.

## Critério de decisão (definido ANTES de ver os resultados)
1. **Menor EER na validação privada** (`private_val.eer`), pois é o domínio da aplicação;
2. desempate: AUC no LFW (benchmark externo — **não** usado como critério principal para evitar
   escolher o modelo pelo conjunto de teste público);
3. custo: tempo de inferência por face (`cost.inference_ms_per_face`) — aceitável até ~100 ms em CPU.

## Evidências até agora (LFW, protocolo 10-fold, CPU)
| Modelo | Acurácia | EER | ms/face | Parâmetros |
|---|---|---|---|---|
| FaceNet (E01) | 99,10% | 0,93% | 61,7 | 23,5 M |
| ArcFace iResNet-50 (E02) | 99,85% | 0,30% | 90,3 | 43,6 M |
| MobileFaceNet (E08) | 99,67% | 0,32% | 5,0 | 3,4 M |

No teste preliminar na pasta do grupo (3 pessoas × 3 fotos) os três empataram (AUC 1,0).

## Decisão
**Aceita (Parte 2):** **MobileFaceNet com ArcFace loss (E08)**, pré-treinado e congelado, via ONNX
(`checkpoints/public/buffalo_s/w600k_mbf.onnx`). É o modelo configurado em `configs/app.yaml`.

Justificativa, seguindo os critérios definidos antes dos resultados:
1. **Critério 1 (domínio do grupo):** no teste da pasta do grupo os três modelos empataram
   (AUC 1,0), então esse critério não separa os candidatos.
2. **Critério 2 (LFW, desempate):** empata com o iResNet-50 dentro do desvio entre folds
   (EER 0,32% × 0,30%; acurácia 99,67% ± 0,20 × 99,85% ± 0,20) e supera a FaceNet
   (EER 0,93%), inclusive a fine-tunada (E03 não melhorou o baseline).
3. **Critério 3 (custo):** **5,0 ms/face** contra 90,3 ms do iResNet-50 e 61,7 ms da FaceNet,
   com 3,4 M parâmetros. Isso é o que permite o reconhecimento **contínuo pela câmera** em CPU
   (spec 007), em que o modelo roda várias vezes por segundo.

Revisão: se o E06/E07 no dataset do grupo ampliado mostrar EER do iResNet-50 claramente menor
(fora do desvio), trocar `model.path` em `configs/app.yaml` e rodar o E07 de novo (ADR-003).

## Consequências
- Se o ArcFace vencer, a aplicação herda a restrição de uso **não comercial** (declarado no README).
- Se uma FaceNet fine-tunada vencer, o checkpoint fica em `checkpoints/` (nunca versionado).
