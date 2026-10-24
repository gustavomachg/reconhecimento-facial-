# Spec 002 — Cadastro de pessoa

**Status:** implementada · **Módulos:** `service.py` (`enroll`), `storage.py`, `app/main.py` (`POST /api/persons`)
**Testes:** `tests/test_service.py`, `tests/test_storage.py`, `tests/test_app.py`

## Fluxo
`nome + 1..N imagens → detecção (spec 001) → alinhamento → embedding L2-normalizado → validações → SQLite`

## Requisitos
1. Nome obrigatório (não vazio); de 1 a `max_enroll_images` (padrão 10) imagens; máx. 10 MB cada.
2. **Cada** imagem deve conter exatamente uma face de qualidade adequada (spec 001); qualquer
   falha aborta o cadastro inteiro — nada é salvo parcialmente.
3. **Consistência:** a menor similaridade cosseno entre duas fotos do mesmo cadastro deve ser
   ≥ `consistency_threshold` (0,40); senão `InconsistentEnrollment` (fotos de pessoas diferentes).
4. **Duplicidade:** se alguma foto tiver cosseno ≥ `duplicate_threshold` (0,75) com qualquer
   embedding já cadastrado → `DuplicatePerson`, informando com quem se parece.
5. Persistir uma linha em `persons` (UUID opaco) e um embedding por foto em `embeddings`.
6. Fotos **não** são guardadas por padrão (`store_photos: false`, minimização LGPD). Se ativado,
   guarda só o recorte alinhado em `faces/<uuid>/`.
7. Log `enroll_ok` com `person_id` e nº de imagens — **sem nome**.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | 3 fotos boas da mesma pessoa | cadastrar "A" | 201; 1 pessoa, 3 embeddings |
| CA2 | foto sem face | cadastrar | 422 `no_face`; nada salvo |
| CA3 | foto com 2 faces | cadastrar | 422 `multiple_faces`; nada salvo |
| CA4 | foto borrada / face pequena | cadastrar | 422 `low_quality`; nada salvo |
| CA5 | "A" já cadastrada | cadastrar nova foto de "A" | 422 `duplicate_person` |
| CA6 | fotos de duas pessoas | cadastrar | 422 `inconsistent_enrollment` |
| CA7 | nome vazio ou sem fotos | cadastrar | 400 |
| CA8 | qualquer cadastro | inspecionar logs | nome não aparece; UUID aparece |
