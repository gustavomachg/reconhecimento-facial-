# ADR-003 — Threshold de reconhecimento

**Status:** proposto — **valor final pendente do E07**
**Data:** 2026-10-24

## Contexto
A decisão "reconhecido × desconhecido" é `score ≥ threshold`. Threshold baixo → desconhecidos
aceitos como alguém cadastrado (FAR alto, falha de segurança); alto → pessoas cadastradas
rejeitadas (FRR alto, falha de usabilidade).

## Regras
1. O threshold é escolhido **apenas na validação** (sessões de validação + desconhecidos de
   validação). O teste é usado **uma vez**, depois, para reportar o desempenho com o valor congelado.
2. Cada matcher tem sua própria escala de score; o threshold vale para o par (modelo, matcher).
3. Estratégia padrão: **EER** (equilíbrio FAR = FRR). Alternativa: **FAR-alvo** (ex.: 1%) se a
   aplicação priorizar não aceitar desconhecidos.

## Decisão
_A preencher após o E07:_ modelo `____` + matcher `____`, estratégia `____`,
threshold = `____`; no teste: FAR desconhecidos = `____`, FRR conhecidos = `____`.

Valor provisório em `configs/app.yaml`: **`0.25`** (MobileFaceNet + protótipo). Ponto de partida:
o threshold de EER do E08 no LFW (0,206; FAR 0,27% / FRR 0,40%), arredondado para cima para
priorizar a rejeição de desconhecidos (a aplicação prefere dizer "Pessoa desconhecida" a confundir
pessoas). Ele vem de um benchmark público e de pares foto × foto, não do domínio da aplicação
nem do matcher protótipo. Por isso continua **provisório** até o E07.
`duplicate_threshold` (0,60) e `consistency_threshold` (0,20) seguem a mesma escala e também
devem ser revistos com o histograma de scores do E07.

Para fechar esta decisão (no computador do grupo, dados privados nunca saem da máquina):
```bash
python -m facerec.compare_matchers --config configs/experiments/E06_matchers_open_set.yaml
python -m facerec.threshold        --config configs/experiments/E07_threshold_sweep.yaml
```
Copiar para cá apenas os valores **agregados** de `experiments/results/E07/metrics.json`.

## Consequências
Trocar de modelo ou matcher exige rodar o E07 de novo.
