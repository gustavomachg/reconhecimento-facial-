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

Valor provisório em `configs/app.yaml`: `0.55` (cosseno/protótipo FaceNet) — **apenas** para a
aplicação funcionar antes dos experimentos; não é resultado experimental.

## Consequências
Trocar de modelo ou matcher exige rodar o E07 de novo.
