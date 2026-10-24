# Dados

Nenhum dado é versionado. `data/public/` e `data/private/` estão no `.gitignore` e o hook
`scripts/check_private_files.py` bloqueia qualquer commit dessas pastas.

## Públicos

| Dataset | Fonte oficial | Como obter | Uso no trabalho |
|---|---|---|---|
| LFW | <http://vis-www.cs.umass.edu/lfw/> | `python scripts/download_lfw.py` | benchmark (pares 10-fold) — nunca treino |
| CelebA | <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html> | manual, ver `scripts/download_celeba.md` | subconjunto de identidades para fine-tuning (E03–E05) |

Depois de baixar, gere as faces alinhadas:

```
python -m facerec.data.preprocess --dataset lfw    --model facenet
python -m facerec.data.preprocess --dataset lfw    --model arcface
python -m facerec.data.preprocess --dataset celeba --model facenet
python scripts/check_identity_overlap.py --model facenet
python scripts/check_duplicates.py --dataset celeba --config configs/experiments/E03_facenet_partial_triplet.yaml --exclude-out
```

## Privado (criado localmente pelo aluno)

Requisitos: só adultos, com **termo de consentimento assinado** (guardado fora do repositório,
ex.: `consent/`, também ignorado). Ver `docs/privacy.md`.

### Estrutura obrigatória

```
data/private/raw/
├── P01/                 # código opaco — NUNCA o nome da pessoa
│   ├── s1/              # sessão 1 (ex.: dia 1, sala, luz natural)  -> galeria/cadastro
│   │   ├── 001.jpg ...
│   ├── s2/              # sessão 2 (outro dia/local/luz)            -> validação
│   └── s3/              # sessão 3                                  -> teste
├── P02/ ...
```

- Mínimo recomendado: **6 pessoas** (4 conhecidas + 1 desconhecida de validação + 1 de teste),
  ~5–10 fotos por sessão. Quanto mais pessoas, mais confiáveis as métricas.
- Variar entre fotos: iluminação, pose (frontal, ¾, leve perfil), distância, expressão,
  fundo, com/sem óculos.
- A tabela que liga `P01` ao nome real (se necessária) fica **fora** do repositório.
- Fotos de celular: renomeie (`IMG_1234.JPG` → `001.jpg`) e evite nomes de pessoas nos arquivos.

> Se você já tem fotos soltas (ex.: na pasta `pessoas/` na raiz, também ignorada pelo git),
> mova-as manualmente para essa estrutura. Uma foto por pessoa basta para **cadastrar na
> aplicação**, mas não para os experimentos E06/E07, que precisam de sessões diferentes.

Depois:

```
python -m facerec.data.preprocess --dataset private --model facenet
python -m facerec.data.preprocess --dataset private --model arcface
python scripts/check_duplicates.py --dataset private --config configs/experiments/E06_matchers_open_set.yaml
```

Os nomes das sessões (`s1`, `s2`, `s3`) e o número de desconhecidos são configuráveis nos YAMLs
(`private.gallery_sessions`, `val_sessions`, `test_sessions`, `n_unknown_val`, `n_unknown_test`).
