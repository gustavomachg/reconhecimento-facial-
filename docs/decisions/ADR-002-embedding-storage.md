# ADR-002 — Armazenamento de pessoas e embeddings

**Status:** aceito · **Data:** 2026-10-24

## Contexto
A aplicação precisa persistir pessoas cadastradas e seus embeddings (dado biométrico sensível),
suportar exclusão completa e funcionar offline, com dezenas (não milhões) de pessoas.

## Opções
| Opção | Prós | Contras |
|---|---|---|
| **SQLite local** | arquivo único, transações, `ON DELETE CASCADE`, `secure_delete`, sem servidor | busca linear (ok para dezenas/centenas de pessoas) |
| Arquivos `.npy` por pessoa | simples | exclusão/consistência manuais, sem transação |
| Banco vetorial (FAISS/Chroma) | busca ANN escalável | dependência extra, exclusão física menos transparente; desnecessário na escala |

## Decisão
SQLite em `embeddings/facerec.db` (ignorado pelo git), tabelas `persons` e `embeddings`
(BLOB float32, um embedding por foto), com:
- **UUID opaco** como chave — é o único identificador usado em logs;
- `PRAGMA foreign_keys=ON` + `ON DELETE CASCADE` — excluir a pessoa apaga os embeddings;
- `PRAGMA secure_delete=ON` + `VACUUM` após exclusão — bytes apagados não ficam no arquivo;
- fotos **não** armazenadas por padrão (minimização); opcionalmente só o recorte alinhado.

A busca é feita em memória pelo matcher (cache reconstruído após cadastro/exclusão).

## Consequências
- Escala linear: se o número de pessoas crescer muito, trocar o matcher por índice ANN sem mudar o esquema.
- Embeddings não são criptografados em repouso; a proteção é o arquivo local + ignorado pelo git.
  Melhoria futura: SQLCipher / criptografia do disco.
