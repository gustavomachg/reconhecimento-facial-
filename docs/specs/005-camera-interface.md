# Spec 005 — Interface web com câmera

**Status:** implementada · **Módulos:** `src/facerec/app/main.py`, `src/facerec/app/static/index.html`
**Testes:** `tests/test_app.py`

## Requisitos
1. Aplicação web **local** (FastAPI + Uvicorn em `127.0.0.1:8000`); nenhuma imagem sai da máquina.
2. Webcam via `getUserMedia`: botão "Ligar câmera", "Reconhecer" (envia o frame atual) e
   "Capturar foto da câmera" (acumula fotos para o cadastro). Também aceita upload de arquivos.
3. Cadastro: campo nome + fotos capturadas/enviadas → `POST /api/persons`.
4. Lista de pessoas cadastradas com botão **Excluir** (confirmação) → `DELETE /api/persons/{id}`.
5. Estado inicial após clonar: **"Nenhuma pessoa cadastrada."**
6. Mensagens de erro de domínio exibidas em linguagem clara (código + mensagem da exceção).

## API
| Método | Rota | Resposta |
|---|---|---|
| GET | `/` | página |
| GET | `/api/persons` | `{persons: [...], message}` |
| POST | `/api/persons` (multipart `name`, `files[]`) | 201 `{id, name, n_embeddings, message}` |
| POST | `/api/recognize` (multipart `file`) | `{recognized, person_id, name, score, threshold, message}` |
| DELETE | `/api/persons/{id}` | 204 · 404 se não existir |
Erros de domínio → 422 `{code, message}`; entrada inválida → 400; arquivo > 10 MB → 413.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | repositório recém-clonado | abrir a página | "Nenhuma pessoa cadastrada." |
| CA2 | câmera ligada | cadastrar com 3 capturas | pessoa aparece na lista |
| CA3 | pessoa cadastrada | "Reconhecer" | "Pessoa reconhecida: X (score)" |
| CA4 | pessoa não cadastrada | "Reconhecer" | "Pessoa desconhecida" |
| CA5 | sem rosto no quadro | "Reconhecer" | "Nenhuma face detectada na imagem." |
