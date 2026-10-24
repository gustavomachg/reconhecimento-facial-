# Spec 004 — Pessoa desconhecida (open-set)

**Status:** implementada · **Módulos:** `matching.py` (`identify`), `service.py`, `threshold.py`
**Testes:** `tests/test_matching.py::test_unknown_rejected_with_threshold`, `tests/test_service.py::test_unknown_person`

## Contexto
O sistema opera em **conjunto aberto**: a maioria das pessoas que aparecem na câmera NÃO está
cadastrada. Um classificador fechado sempre escolheria "alguém"; por isso a decisão final é
sempre por threshold sobre o score do melhor candidato.

## Requisitos
1. Se `score < threshold` → **"Pessoa desconhecida"**, sem revelar o candidato mais próximo.
2. O threshold é escolhido **somente na validação** (E07), por EER ou FAR-alvo; o teste é usado
   uma única vez para reportar FAR/FRR finais (ADR-003, ADR-004).
3. Desconhecidos de validação e de teste são pessoas diferentes entre si e nunca estão na galeria.
4. Log `recognize_unknown` só com score e latência.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | "A" e "B" cadastradas | reconhecer pessoa "C" | "Pessoa desconhecida" |
| CA2 | "A" cadastrada e depois excluída | reconhecer "A" | "Pessoa desconhecida" |
| CA3 | E07 executado | ler `experiments/results/E07/metrics.json` | FAR de desconhecidos e FRR de conhecidos no teste |
