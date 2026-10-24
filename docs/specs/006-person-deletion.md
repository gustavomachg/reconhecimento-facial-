# Spec 006 — Exclusão de pessoa

**Status:** implementada · **Módulos:** `storage.py` (`delete_person`), `service.py` (`delete`)
**Testes:** `tests/test_storage.py::test_delete_removes_person_embeddings_and_photos`,
`tests/test_storage.py::test_deleted_bytes_not_left_in_db_file`,
`tests/test_service.py::test_delete_removes_everything_and_person_becomes_unknown`

## Contexto
Direito do titular à eliminação (LGPD art. 18, VI). A exclusão precisa ser **completa** e
**verificável**.

## Requisitos
1. Remover a linha em `persons` e, por `ON DELETE CASCADE`, **todos** os embeddings.
2. Remover fotos armazenadas (`faces/<uuid>/`), se existirem.
3. `PRAGMA secure_delete=ON` + `VACUUM`: os bytes apagados não permanecem no arquivo `.db`.
4. Invalidar o cache do matcher: a pessoa passa a ser "desconhecida" imediatamente.
5. ID inexistente → `PersonNotFound` (404).
6. Log `delete_ok` apenas com o UUID.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | "A" e "B" cadastradas | excluir "A" | 0 linhas de "A" em `persons` e `embeddings` (consulta direta ao SQLite) |
| CA2 | "A" com fotos armazenadas | excluir "A" | pasta `faces/<uuid>` removida |
| CA3 | "A" excluída | procurar bytes do embedding/nome no arquivo .db | não encontrados |
| CA4 | "A" excluída | reconhecer foto de "A" | "Pessoa desconhecida" |
| CA5 | "B" cadastrada | excluir "A" | "B" intacta |
| CA6 | id inexistente | excluir | 404 |

## Eliminação total
Ao fim da disciplina: `python scripts/purge_private_data.py --yes` (ver `docs/privacy.md`).
