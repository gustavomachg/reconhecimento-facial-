# Reconhecedor Facial — Trabalho de IA (UFMS)

[![CI](https://github.com/gustavomachg/reconhecimento-facial-/actions/workflows/ci.yml/badge.svg)](https://github.com/gustavomachg/reconhecimento-facial-/actions/workflows/ci.yml)

Sistema de reconhecimento facial em duas partes:

1. **Experimentação** — comparar modelos (FaceNet × ArcFace), estratégias de fine-tuning
   (parcial × completo), funções de perda (Triplet × ArcFace), comparadores e threshold.
2. **Aplicação** — web local com webcam: cadastro, reconhecimento (conhecido/desconhecido) e exclusão.

```
Imagem → Detecção (MTCNN) → Alinhamento → Rede neural → Embedding (L2) → Comparação → Threshold
       → "Pessoa reconhecida: X (score)"  |  "Pessoa desconhecida"
```

Metodologia: **Spec-Driven Development** — specs em [`docs/specs/`](docs/specs), decisões em
[`docs/decisions/`](docs/decisions), privacidade em [`docs/privacy.md`](docs/privacy.md).

**Parte 1 (foco atual):** protocolo de cada experimento em [`docs/experiments.md`](docs/experiments.md),
rastreabilidade enunciado → código em [`docs/parte1_checklist.md`](docs/parte1_checklist.md) e
roteiro do vídeo em [`docs/video_parte1.md`](docs/video_parte1.md).

---

## Privacidade (LGPD)

Dados biométricos são dados pessoais sensíveis. Este repositório **não contém** fotos, embeddings,
bancos, checkpoints treinados com dados privados nem termos de consentimento. Proteções:
`.gitignore`, hook de pre-commit [`scripts/check_private_files.py`](scripts/check_private_files.py),
logs só com UUIDs opacos, exclusão completa verificada por teste, resultados privados apenas
agregados e eliminação final com [`scripts/purge_private_data.py`](scripts/purge_private_data.py).
Detalhes em [`docs/privacy.md`](docs/privacy.md).

## Datasets

| Dataset | Fonte oficial | Finalidade | Licença / termos | Forma de uso |
|---|---|---|---|---|
| **LFW** | <http://vis-www.cs.umass.edu/lfw/> | benchmark de verificação (6000 pares, 10 folds) | disponibilizado para pesquisa; não redistribuído aqui | `scripts/download_lfw.py` |
| **CelebA** | <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html> | fine-tuning (subconjunto de identidades com ≥15 imagens) | somente pesquisa não comercial; **proibida a redistribuição** | download manual — [`scripts/download_celeba.md`](scripts/download_celeba.md) |
| **Privado** | coletado pelo aluno | domínio da aplicação; E06/E07; validação de E01–E05 | consentimento LGPD; só adultos | local, [`data/README.md`](data/README.md) |

Pesos pré-treinados (baixados das fontes oficiais para `checkpoints/public/` / cache do torch):

| Modelo | Fonte | Observação |
|---|---|---|
| InceptionResnetV1 (VGGFace2) | `facenet-pytorch` (timesler) | baixado automaticamente |
| ArcFace iResNet-50 `w600k_r50` (buffalo_l) | `insightface` (deepinsight) | **uso não comercial / acadêmico apenas** — `scripts/download_arcface.py` |
| ArcFace MobileFaceNet `w600k_mbf` (buffalo_s) | `insightface` (deepinsight) | **uso não comercial / acadêmico apenas** — `scripts/download_arcface.py --pack buffalo_s` |

Sobreposição CelebA × LFW: o CelebA não publica nomes, então ela é verificada empiricamente por
embeddings (`scripts/check_identity_overlap.py`) e as identidades suspeitas são excluídas do treino
(ADR-004). Resultado agregado em `experiments/results/identity_overlap.json`.

## Arquitetura do código

| Camada | Módulo | Responsabilidade |
|---|---|---|
| Detecção | `src/facerec/detection.py` | MTCNN, qualidade (tamanho, nitidez), alinhamento `crop`/`similarity` |
| Modelo | `src/facerec/models/` | `FaceNetEmbedder` (PyTorch), `ArcFaceEmbedder` (ONNX) — embeddings L2 |
| Treino | `src/facerec/train.py`, `losses.py` | fine-tuning parcial/completo; Triplet batch-hard; ArcFace |
| Dados | `src/facerec/data/` | leitura, splits sem vazamento, augmentation, PK sampler, dHash |
| Avaliação | `src/facerec/evaluate.py`, `metrics.py` | LFW 10-fold, ROC/AUC, EER, FAR/FRR, F1, custo |
| Matching | `src/facerec/matching.py` | cosseno, euclidiana, protótipo, k-NN, SVM (open-set) |
| Threshold | `src/facerec/threshold.py` | EER, FAR-alvo, máx. F1/acurácia — sempre na validação |
| Storage | `src/facerec/storage.py` | SQLite com UUID opaco, cascade, secure_delete |
| Serviço | `src/facerec/service.py` | casos de uso `enroll`, `recognize`, `delete` |
| Interface | `src/facerec/app/` | FastAPI + página com webcam |

Exceções de domínio: `NoFaceDetected`, `MultipleFacesDetected`, `LowQualityImage`, `InvalidImage`,
`DuplicatePerson`, `InconsistentEnrollment`, `PersonNotFound`.

## Instalação

Requisitos: **Python 3.11 ou 3.12** (o `facenet-pytorch` 2.6 exige torch ≤ 2.3 e numpy < 2, que
não existem para 3.13+), Git e ~3 GB livres. GPU NVIDIA é opcional (só acelera o fine-tuning).
Testado em Windows 11 e Linux (Ubuntu, via CI).

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1      # se bloquear: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

**Linux / macOS**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

Com GPU NVIDIA, troque o índice do PyTorch por `https://download.pytorch.org/whl/cu121`.

> **Windows — caminho curto:** o Windows limita caminhos a 260 caracteres, e algumas bibliotecas
> (scikit-learn, torch) têm arquivos internos com nomes longos. Clone o repositório numa pasta
> curta (ex.: `C:\projetos\reconhecedor-facial`) ou ative caminhos longos
> (`LongPathsEnabled`). O erro típico é `DLL load failed ... nome do arquivo ou a extensão é muito grande`.
Para desenvolver: `pre-commit install` (ativa o bloqueio de dados privados nos commits).

## Verificação rápida (para o avaliador, ~2 min, sem baixar datasets)

```bash
pytest
```

São mais de 100 testes, **todos com dados sintéticos** (o repositório não contém nem baixa dados
pessoais). Eles incluem um teste de ponta a ponta (`tests/test_pipeline_smoke.py`) que usa os YAMLs
reais dos experimentos e roda, com imagens geradas na hora:

- fine-tuning parcial com Triplet (E03) e com ArcFace (E04);
- avaliação pelo protocolo LFW 10-fold e na validação privada (E01);
- comparadores em conjunto aberto (E06) e varredura de threshold com teste único (E07).

O mesmo roda a cada push no GitHub Actions (Linux e Windows, Python 3.11 e 3.12) —
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Para um experimento real com dado público: o E01 só precisa do LFW (download automático, ~180 MB):

```bash
python scripts/download_lfw.py
python -m facerec.data.preprocess --dataset lfw --model facenet
python -m facerec.evaluate --config configs/experiments/E01_facenet_baseline.yaml
```

O dataset privado não acompanha o repositório (LGPD); os experimentos que dependem dele
(validação privada, E06, E07) pulam essa parte com um aviso ou explicam como criá-lo
([`data/README.md`](data/README.md)).

## Avaliação no dataset do grupo (pasta `pessoas/`)

O dataset principal do trabalho é a pasta **`pessoas/`**, montada localmente pelo grupo (ignorada
pelo git, nunca vai para o GitHub). Uma subpasta por pessoa:

```
pessoas/
├── pessoa_a/   foto1.jpg, foto2.jpg, ...   (mín. 2 fotos; recomendado 5+ com variações)
├── pessoa_b/   ...
└── ...         (mín. 2 pessoas; quanto mais, mais confiáveis as métricas)
```

```bash
python -m facerec.evaluate_folder                  # detecta e alinha com MTCNN
python -m facerec.evaluate_folder --pre-aligned    # se as fotos já forem recortes do rosto
```

O comando compara os modelos (FaceNet, ArcFace iResNet-50 e MobileFaceNet) e os comparadores
(cosseno, protótipo, k-NN) e responde se cada pessoa é **reconhecida** e se pessoas fora do
cadastro são **rejeitadas como desconhecidas**:

- **conhecidos:** cada foto sai da galeria e é consultada contra as outras (a pessoa continua
  cadastrada com as demais fotos) — deve ser reconhecida;
- **desconhecidos:** cada pessoa sai inteira da galeria e suas fotos são consultadas — devem dar
  "Pessoa desconhecida";
- **threshold sem vazamento:** escolhido por validação cruzada em grupos de pessoas (o threshold
  usado nas fotos de uma pessoa nunca é calculado com as fotos dela).

Saída **agregada e anônima** (nenhum nome de pasta ou arquivo) em `experiments/results/dataset_grupo/`:
`results_table.md`, `metrics.json` (acurácia, precision, recall, F1, EER, ROC-AUC, FAR, FRR,
matriz de confusão, custo), `hist_<modelo>.png`, `roc_comparison.png`, `confusion_*.png`.

### Reconhecer fotos usando a pasta `pessoas/` como cadastro

```bash
python -m facerec.recognize foto_nova.jpg     # uma foto
python -m facerec.recognize pasta_de_teste/   # várias fotos
python -m facerec.recognize pessoas/          # cada foto do cadastro contra as OUTRAS (acertos/erros)
```

Usa o modelo, o comparador e o threshold de `configs/app.yaml` e responde, para cada foto,
"Pessoa reconhecida: X (score)" ou "Pessoa desconhecida". Para cadastrar a pasta na aplicação
web: `python scripts/import_people.py`.

## Reprodução dos experimentos (Parte 1)

```bash
# 1. dados
python scripts/download_lfw.py
python scripts/download_arcface.py
python scripts/download_arcface.py --pack buffalo_s
#    CelebA: manual (scripts/download_celeba.md); privado: data/README.md
python -m facerec.data.preprocess --dataset lfw     --model facenet
python -m facerec.data.preprocess --dataset lfw     --model arcface
python -m facerec.data.preprocess --dataset celeba  --model facenet
python -m facerec.data.preprocess --dataset private --model facenet
python -m facerec.data.preprocess --dataset private --model arcface
python scripts/check_identity_overlap.py --model facenet
python scripts/check_duplicates.py --dataset celeba  --config configs/experiments/E03_facenet_partial_triplet.yaml --exclude-out
python scripts/check_duplicates.py --dataset private --config configs/experiments/E06_matchers_open_set.yaml

# 2. modelos pré-treinados
python -m facerec.evaluate --config configs/experiments/E01_facenet_baseline.yaml
python -m facerec.evaluate --config configs/experiments/E02_arcface_pretrained.yaml
python -m facerec.evaluate --config configs/experiments/E08_mobilefacenet_pretrained.yaml

# 3. fine-tuning (treino + avaliação)
python -m facerec.train    --config configs/experiments/E03_facenet_partial_triplet.yaml
python -m facerec.evaluate --config configs/experiments/E03_facenet_partial_triplet.yaml
python -m facerec.train    --config configs/experiments/E04_facenet_partial_arcface.yaml
python -m facerec.evaluate --config configs/experiments/E04_facenet_partial_arcface.yaml
python -m facerec.train    --config configs/experiments/E05_facenet_full_finetune.yaml
python -m facerec.evaluate --config configs/experiments/E05_facenet_full_finetune.yaml

# 4. escolher o vencedor (ADR-001) e ajustar `model` em E06/E07, depois:
python -m facerec.compare_matchers --config configs/experiments/E06_matchers_open_set.yaml
python -m facerec.threshold        --config configs/experiments/E07_threshold_sweep.yaml

# 5. tabela e ROC comparativa
python -m facerec.report
```

Cada YAML registra hipótese, dataset, modelo, estratégia, loss, hiperparâmetros, seed, épocas,
optimizer, LR, batch size e transforms; versões e hardware vão para `metrics.json`/`train_summary.json`.

### Experimentos

| ID | Modelo | Estratégia | Hipótese |
|---|---|---|---|
| E01 | FaceNet (VGGFace2) | pré-treinado, cosseno | baseline |
| E02 | ArcFace (insightface) | pré-treinado, cosseno | margem angular → embeddings mais separáveis que E01 |
| E03 | FaceNet | fine-tuning parcial (`repeat_3`, `block8`, `last_*`) + Triplet (batch-hard) | adapta ao domínio sem overfitting |
| E04 | FaceNet | fine-tuning parcial + ArcFace loss | converge melhor que Triplet com poucos dados |
| E05 | FaceNet | fine-tuning completo (ArcFace loss, LR 10× menor) | ganho menor/overfitting vs. parcial |
| E06 | melhor de E01–E05 | cosseno × euclidiana × k-NN × SVM × protótipos (open-set, validação) | escolha do comparador |
| E07 | configuração final | varredura de threshold na validação + teste único | threshold ótimo (EER ou FAR-alvo) |
| E08 | ArcFace MobileFaceNet (insightface) | pré-treinado, cosseno | backbone leve perde pouco desempenho e é bem mais rápido que o E02 (custo × desempenho) |

## Resultados

Hardware e ambiente dos resultados abaixo: CPU Intel (12 threads), **sem GPU**, Windows 11,
Python 3.11.17, torch 2.2.2+cpu, onnxruntime 1.17.3, facenet-pytorch 2.6.0 (registrado em
`environment` de cada `metrics.json`). Tabela gerada por `python -m facerec.report`.

### Dataset público — LFW (protocolo oficial: 6000 pares, 10 folds)

O threshold de cada fold é escolhido nos outros 9 folds; a acurácia é a média ± desvio dos 10 folds.

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | EER | ROC-AUC | FAR | FRR | ms/face (CPU) | Parâmetros |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E01 | FaceNet (VGGFace2) | pré-treinado, cosseno | LFW | 99,10% ± 0,43 | 99,10% | 0,93% | 0,9991 | 0,97% | 0,83% | 61,7 | 23,5 M |
| E02 | ArcFace iResNet-50 (WebFace600K) | pré-treinado, cosseno | LFW | **99,85% ± 0,20** | **99,85%** | **0,30%** | **0,9995** | **0,00%** | **0,30%** | 90,3 | 43,6 M |
| E08 | MobileFaceNet, ArcFace loss (WebFace600K) | pré-treinado, cosseno | LFW | 99,67% ± 0,20 | 99,67% | 0,32% | 0,9993 | 0,27% | 0,40% | **5,0** | **3,4 M** |

Matrizes de confusão (soma dos 10 folds, 3000 pares de cada tipo):

| Modelo | Mesma pessoa aceita (VP) | Mesma pessoa rejeitada (FN) | Diferentes aceitas (FP) | Diferentes rejeitadas (VN) |
|---|---|---|---|---|
| FaceNet | 2975 | 25 | 29 | 2971 |
| ArcFace iResNet-50 | 2991 | 9 | 0 | 3000 |
| MobileFaceNet | 2988 | 12 | 8 | 2992 |

Gráficos: [`roc_comparison_lfw.png`](experiments/results/roc_comparison_lfw.png) e, em
`experiments/results/E0x/`, `roc_lfw.png`, `hist_lfw.png` (mesma pessoa × diferentes) e `confusion_lfw.png`.

### Impacto do pré-processamento (ablação: escolha da face)

1812 das 13 233 imagens do LFW têm mais de um rosto. Na primeira execução o pré-processamento
escolhia o **maior** rosto; no LFW a pessoa rotulada é a do **centro** da imagem, e o maior rosto às
vezes é outra pessoa. Com isso, pares "mesma pessoa" viravam, na prática, pessoas diferentes.

| Modelo | Acurácia (maior face) | Acurácia (face central) | EER (maior) | EER (central) | Pares "mesma pessoa" rejeitados |
|---|---|---|---|---|---|
| FaceNet | 95,90% | 99,10% | 6,10% | 0,93% | 219 → 25 |
| ArcFace iResNet-50 | 96,42% | 99,85% | 6,02% | 0,30% | 215 → 9 |
| MobileFaceNet | 96,28% | 99,67% | 6,30% | 0,32% | 216 → 12 |

O sinal do problema: três modelos muito diferentes rejeitavam quase os **mesmos** ~216 pares e
quase nunca aceitavam impostores — o erro estava nos dados de entrada, não nos modelos.
Resultados "antes" em [`experiments/results/ablation_face_selection/`](experiments/results/ablation_face_selection).

### Dataset do grupo (pasta `pessoas/`)

Avaliado com `python -m facerec.evaluate_folder` (protocolo na seção "Avaliação no dataset do
grupo"). Resultados a publicar em `experiments/results/dataset_grupo/` quando o dataset for ampliado
(o teste preliminar com 3 pessoas × 3 fotos foi pequeno demais para diferenciar os modelos:
todos atingiram AUC 1,0).

### Experimentos ainda não executados

| Experimento | Motivo |
|---|---|
| E03, E04, E05 (fine-tuning) | exigem o CelebA (download manual) e GPU para tempo razoável |
| E06, E07 (comparadores e threshold, protocolo por sessões) | exigem o dataset privado organizado por sessão; o `evaluate_folder` já compara comparadores e calibra o threshold na pasta do grupo |

### Análise

- **Melhor modelo no benchmark público:** o **ArcFace iResNet-50 (E02)** teve a maior acurácia
  (99,85%), o menor EER (0,30%) e **nenhum falso aceite** em 3000 pares de pessoas diferentes.
  A hipótese do E02 se confirma: a margem angular aditiva (e o pré-treino em mais identidades) gera
  embeddings mais separáveis que a FaceNet/Triplet do E01 (EER 0,93%, ~3× maior).
- **Custo × desempenho (E02 × E08):** com a mesma loss e os mesmos dados de pré-treino, o
  MobileFaceNet perde só 0,18 ponto de acurácia (EER 0,32% × 0,30%), mas é **~18× mais rápido**
  (5,0 × 90,3 ms/face em CPU) e **~13× menor** (3,4 M × 43,6 M parâmetros). Para uma aplicação em
  tempo real em notebook sem GPU, é a melhor relação custo × desempenho; a diferença de acurácia
  está dentro do desvio entre folds (±0,20).
- **Impacto do pré-processamento:** escolher a face errada em imagens com várias pessoas custava
  3,2–3,4 pontos de acurácia e multiplicava o EER por ~7–20× — mais do que a diferença entre os modelos.
  Detecção/alinhamento corretos importam tanto quanto a arquitetura.
- **Erros restantes:** com a face correta, os erros são raros e dominados por pares "mesma pessoa"
  rejeitados (FN), tipicamente com grande diferença de pose, idade ou iluminação entre as fotos.
- **Impacto do fine-tuning / Triplet × ArcFace / parcial × completo:** _pendente (E03–E05)._
- **Threshold:** no LFW, o threshold é calibrado por fold e depende do modelo (cosseno médio: FaceNet
  0,37; ArcFace 0,25; MobileFaceNet 0,21) — por isso cada modelo precisa do seu próprio threshold; na
  pasta do grupo, por validação cruzada entre pessoas (`threshold_mean` em `metrics.json`).

### Limitações conhecidas
- Dataset privado pequeno → alta variância das métricas privadas.
- O pré-treino (VGGFace2, WebFace600K) pode conter identidades do LFW — não controlável; o LFW
  serve como referência externa, não como critério de escolha.
- Sem anti-spoofing: uma foto da pessoa na tela pode ser reconhecida.
- Os modelos ArcFace (iResNet-50 e MobileFaceNet) são usados congelados (ONNX); apenas a FaceNet é fine-tunada.
- Viés demográfico dos datasets públicos (celebridades) pode afetar o desempenho.

## Aplicação (Parte 2)

```bash
uvicorn facerec.app.main:app --reload     # http://127.0.0.1:8000
```

Configuração em [`configs/app.yaml`](configs/app.yaml) — modelo, matcher, threshold, caminhos, device.
Estado inicial após clonar: **"Nenhuma pessoa cadastrada."**

| Funcionalidade | Spec | Comportamento |
|---|---|---|
| Cadastro | [002](docs/specs/002-person-enrollment.md) | nome + fotos (webcam ou arquivo); rejeita sem face, várias faces, baixa qualidade, fotos inconsistentes e duplicatas |
| Reconhecimento | [003](docs/specs/003-face-recognition.md) / [004](docs/specs/004-unknown-person.md) | "Pessoa reconhecida: X (score)" ou "Pessoa desconhecida" |
| Exclusão | [006](docs/specs/006-person-deletion.md) | remove pessoa, embeddings e fotos; `secure_delete` + `VACUUM` |
| Interface | [005](docs/specs/005-camera-interface.md) | web local com webcam |

## Testes

```bash
pytest          # 100% dados sintéticos (embeddings aleatórios, imagens geradas, detector falso)
ruff check . && ruff format --check .
python scripts/check_private_files.py --history   # nenhum dado privado no histórico do git
```

## Eliminação dos dados ao fim da disciplina

```bash
python scripts/purge_private_data.py          # simulação
python scripts/purge_private_data.py --yes    # apaga
```
