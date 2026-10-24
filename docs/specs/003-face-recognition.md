# Spec 003 — Reconhecimento de pessoa cadastrada

**Status:** implementada · **Módulos:** `service.py` (`recognize`), `matching.py`, `app/main.py` (`POST /api/recognize`)
**Testes:** `tests/test_service.py`, `tests/test_matching.py`, `tests/test_app.py`

## Fluxo
`imagem (upload ou frame da webcam) → detecção → embedding → matcher → score → threshold`

## Requisitos
1. Imagem deve ter exatamente uma face (spec 001); erros retornam 422 com código.
2. O matcher (`recognition.matcher`, vencedor do E06) compara o embedding com a galeria e devolve
   a identidade de maior score. Opções: `cosine`, `euclidean`, `prototype`, `knn`, `svm`.
3. Se `score ≥ recognition.threshold` (valor do E07, ADR-003): resposta
   **"Pessoa reconhecida: <nome> (<score>)"**.
4. O matcher é reconstruído (cache invalidado) após cada cadastro ou exclusão.
5. Log `recognize_ok` com `person_id`, score e latência — sem nome.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | "A" cadastrada | reconhecer foto nova de "A" | `recognized=true`, nome "A", score ≥ threshold |
| CA2 | galeria vazia | reconhecer | mensagem "Nenhuma pessoa cadastrada." |
| CA3 | foto sem face / 2 faces | reconhecer | 422 `no_face` / `multiple_faces` |
| CA4 | qualquer | reconhecer | resposta inclui score e threshold usado |
