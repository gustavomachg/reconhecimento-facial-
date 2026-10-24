# ADR-004 — Splits de dados e prevenção de data leakage

**Status:** aceito · **Data:** 2026-10-24

## Decisões

### Datasets públicos
- **CelebA (fine-tuning):** split por **identidade** (70/15/15, seed 42) — treino, validação e
  teste com identidades **disjuntas** (`split_by_identity`, verificado por `assert_disjoint`).
  O split usado é salvo em `experiments/results/<ID>/split.json` (IDs numéricos públicos).
- **LFW (benchmark):** protocolo oficial `pairs.txt`, 10 folds; o threshold de cada fold é
  escolhido nos outros 9. O LFW **nunca** é usado para treinar.
- **Sobreposição CelebA × LFW:** o CelebA não publica nomes, então `scripts/check_identity_overlap.py`
  compara centroides de embeddings; identidades CelebA suspeitas (cosseno ≥ 0,80 com alguma do LFW)
  vão para `data/public/celeba_exclude.txt` e são removidas do treino.
- **Pré-treino:** o VGGFace2 (FaceNet) e o WebFace600K (ArcFace) podem conter celebridades do LFW;
  isso não é controlável por nós e é declarado como limitação no README.

### Dataset privado
- Estrutura `data/private/raw/<P01>/<s1|s2|s3>/` — ID opaco por pessoa e uma pasta por **sessão**
  de captura (dias/locais/iluminação diferentes).
- **Split por sessão**: `s1` = galeria (cadastro), `s2` = validação, `s3` = teste. Fotos da mesma
  sessão nunca são separadas entre conjuntos (seriam quase duplicatas).
- **Open-set**: 1 identidade reservada como desconhecida na validação e **outra** no teste; elas
  nunca entram na galeria.

### Regras gerais
- **Augmentation** só no conjunto de treino e só depois do split (`build_transforms(train=True)`).
- **Quase-duplicatas**: `scripts/check_duplicates.py` (dHash, Hamming ≤ 4) entre conjuntos do
  mesmo split; a cópia do conjunto de avaliação vai para uma lista de exclusão.
- **Seleção de modelo** (E01–E05) por EER na **validação** privada; **comparador** (E06) na
  validação; **threshold** (E07) na validação. O **teste** privado é avaliado uma única vez (E07).

## Consequências
Com poucos participantes, as métricas privadas têm variância alta; reportar o nº de consultas
e tratar diferenças pequenas entre modelos com cautela.
