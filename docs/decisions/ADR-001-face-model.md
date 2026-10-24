# ADR-001 — Modelo de reconhecimento facial

**Status:** proposto — **decisão final pendente dos resultados de E01–E05 e E08**
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

## Decisão
_A preencher após rodar E01–E05 e E08:_ modelo = `____`, EER val. = `____`, motivo: `____`.
Atualizar `configs/app.yaml` (`model`), `E06`/`E07` (`model`, `aligned_root`).

## Consequências
- Se o ArcFace vencer, a aplicação herda a restrição de uso **não comercial** (declarado no README).
- Se uma FaceNet fine-tunada vencer, o checkpoint fica em `checkpoints/` (nunca versionado).
