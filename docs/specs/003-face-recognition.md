# Spec 003 — Reconhecimento de pessoa cadastrada

**Status:** implementada · **Módulos:** `service.py` (`_identify`, `verify_face`, `recognize`), `matching.py`, `app/main.py` (`POST /api/me/door/face` → `verify_face`)
**Testes:** `tests/test_service.py`, `tests/test_matching.py`, `tests/test_app.py`

## Fluxo
`imagem (upload ou frame da webcam) → detecção → embedding → matcher → score → threshold`

## Requisitos
1. Imagem deve ter exatamente uma face (spec 001); erros retornam 422 com código.
2. O matcher (`recognition.matcher`, vencedor do E06) compara o embedding com a galeria e devolve
   a identidade de maior score. Opções: `cosine`, `euclidean`, `prototype`, `knn`, `svm`.
3. Se `score ≥ recognition.threshold` (valor do E07, ADR-003) a pessoa é **reconhecida**.
   Na aplicação (`verify_face`), a identificação é 1:N contra toda a galeria e o resultado é
   comparado com a conta logada: se for o dono, "Rosto reconhecido. Porta liberada."; se for
   outra pessoa cadastrada, "não corresponde à sua conta" (sem revelar quem é) — spec 009.
   O caso de uso `recognize` (usado pela CLI `python -m facerec.recognize` e pelos testes)
   devolve **"Pessoa reconhecida: <nome> (<score>)"**.
4. O matcher é reconstruído (cache invalidado) após cada cadastro ou exclusão.
5. Log `recognize_ok` com `person_id`, score e latência — sem nome.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | "A" cadastrada | reconhecer foto nova de "A" | `recognized=true`, nome "A", score ≥ threshold |
| CA2 | galeria vazia | reconhecer | mensagem "Nenhuma pessoa cadastrada." |
| CA3 | foto sem face / 2 faces | reconhecer | 422 `no_face` / `multiple_faces` |
| CA4 | qualquer | reconhecer | resposta inclui score e threshold usado |
| CA5 | "A" e "B" cadastradas, "A" logada | enviar rosto de "B" | `matched=false`, `reason=other_person`, sem dados de "B" |
