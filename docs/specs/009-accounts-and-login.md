# Spec 009 — Contas: cadastrar-se, entrar e página do usuário

**Status:** implementada · **Módulos:** `auth.py`, `service.py` (`enroll`, `authenticate`,
`verify_face`, `update_profile`, `change_password`), `storage.py`, `app/main.py`,
`app/static/index.html` · **Decisão:** [ADR-005](../decisions/ADR-005-accounts-and-sessions.md)
**Testes:** `tests/test_auth.py`, `tests/test_app.py`, `tests/test_service.py`, `tests/test_storage.py`

## Fluxo
```
Início ──▶ "Entrar" ──▶ e-mail + senha ───────────────────────────┐
   │                                                              ▼
   └─────▶ "Cadastrar-se" ─▶ 1 Rosto ─▶ 2 Dados ─▶ 3 Confirmar ─▶ Conta
                                                                  ├─ Porta   (deslizar · câmera)
                                                                  └─ Usuário (dados · senha · excluir)
```

## Requisitos
### Início
1. A primeira página pergunta **Entrar** ou **Cadastrar-se** e mostra quantas pessoas há no sistema
   (em repositório recém-clonado: **"Nenhuma pessoa cadastrada."**).
2. Quem já tem sessão válida vai direto para a conta; sem sessão, as páginas da conta redirecionam
   para o login.

### Cadastrar-se (3 etapas)
3. **Etapa 1 — Rosto:** a câmera liga sozinha; "Escanear rosto" tira 5 fotos em sequência (spec 008).
   Só avança com ao menos uma foto.
4. **Etapa 2 — Dados:** nome, apartamento/bloco (opcional), morador ou visitante, e-mail e senha
   (mínimo 8 caracteres, digitada duas vezes).
5. **Etapa 3 — Confirmar:** resumo dos dados e **autorização obrigatória** para o uso da imagem
   (LGPD, §14). É a própria pessoa que consente no próprio cadastro.
6. `POST /api/auth/register` cria a pessoa, os embeddings e a conta, e já inicia a sessão.
   Erros levam de volta à etapa certa: problema no rosto → etapa 1; e-mail/senha → etapa 2.
7. E-mail é único (409 `email_in_use`). Rosto já cadastrado → 422 `duplicate_person`, **sem dizer
   de quem é** o cadastro existente.

### Entrar e sair
8. `POST /api/auth/login` com e-mail (sem diferenciar maiúsculas) e senha → sessão por cookie
   `HttpOnly` + `SameSite=Strict`. `POST /api/auth/logout` encerra.
9. E-mail inexistente e senha errada dão a **mesma resposta** (401 "E-mail ou senha incorretos.").
10. Após 5 erros seguidos no mesmo e-mail: 429 por 60 s, mesmo com a senha certa.

### Página principal (Porta)
11. Mostra "Olá, <nome>", o tipo (morador/visitante) e o apartamento.
12. **Deslizar para abrir:** arrastar o botão até o fim chama `POST /api/me/door/open`. Soltar antes
    do fim não abre. Também funciona pelo teclado (Enter).
13. **Ícone de câmera:** abre a câmera e envia quadros a `POST /api/me/door/face` até reconhecer
    (no máximo 10 tentativas). A porta só abre se o rosto for **o do dono da conta**.
14. Rosto de outra pessoa cadastrada → "não corresponde à sua conta", sem revelar quem é.
    Rosto não cadastrado → "Pessoa não cadastrada". Em ambos a porta continua trancada.

### Aba Usuário
15. Mostra o nome, o tipo, o apartamento e o e-mail.
16. **Informações pessoais:** editar nome, apartamento, tipo e e-mail (`PUT /api/me`).
    Trocar o e-mail exige a senha atual. O rosto cadastrado não muda.
17. **Alterar senha** (`POST /api/me/password`): exige a senha atual e encerra as outras sessões.
18. **Excluir minha conta** (`DELETE /api/me`): remove cadastro, e-mail, senha, embeddings e fotos
    (spec 006) e encerra todas as sessões. E-mail e rosto ficam livres para novo cadastro.

### Privacidade e segurança
19. Toda rota `/api/me…` exige sessão (401 `not_authenticated`). Não há rota que liste pessoas
    nem que devolva dados de terceiros.
20. A senha nunca é guardada nem devolvida: só o hash PBKDF2. E-mail, senha e nome não vão para logs.
21. Respostas da API levam `Cache-Control: no-store`.

## API
| Método | Rota | Sessão | Resposta |
|---|---|---|---|
| GET | `/api/status` | não | `{model, matcher, threshold, door_seconds, n_persons, message}` |
| POST | `/api/auth/register` (multipart `name`, `email`, `password`, `unit`, `kind`, `consent`, `files[]`) | não | 201 perfil + cookie |
| POST | `/api/auth/login` (`email`, `password`) | não | perfil + cookie · 401 · 429 |
| POST | `/api/auth/logout` | — | 204 |
| GET | `/api/me` | sim | `{id, name, unit, kind, email, n_embeddings, created_at}` |
| PUT | `/api/me` (`name`, `unit`, `kind`, `email`, `current_password`) | sim | perfil atualizado |
| POST | `/api/me/password` (`current_password`, `new_password`) | sim | `{message}` |
| DELETE | `/api/me` | sim | 204 |
| POST | `/api/me/door/open` | sim | `{door_open: true, door_seconds, method: "manual"}` |
| POST | `/api/me/door/face` (multipart `file`) | sim | `{matched, reason, score, threshold, message, box, door_open, door_seconds}` |

Erros: validação → 400; não autenticado/credenciais → 401; e-mail em uso → 409; rosto (sem face,
várias faces, qualidade, duplicado) → 422; bloqueio de login → 429; inesperado → 500 (JSON).

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | repositório recém-clonado | abrir o site | escolhas "Entrar" / "Cadastrar-se" e "Nenhuma pessoa cadastrada." |
| CA2 | etapa 1 sem foto | — | botão "Próximo" desativado |
| CA3 | etapa 2 com senhas diferentes | "Próximo" | aviso "As senhas não são iguais." |
| CA4 | etapa 3 sem marcar a autorização | "Concluir cadastro" | aviso; nada é enviado; a API também recusa (400) |
| CA5 | cadastro válido | "Concluir cadastro" | 201, sessão iniciada, cai na página da Porta com "Olá, <nome>" |
| CA6 | e-mail já usado | concluir | volta à etapa 2 com "Este e-mail já está cadastrado" |
| CA7 | foto sem rosto no cadastro | concluir | volta à etapa 1 com "Nenhuma face detectada na imagem." |
| CA8 | conta existente | entrar com e-mail e senha | página da Porta |
| CA9 | senha errada ou e-mail inexistente | entrar | mesma mensagem "E-mail ou senha incorretos." |
| CA10 | 5 senhas erradas | 6ª tentativa | "Muitas tentativas…" (429) |
| CA11 | logado | deslizar até o fim | PORTA ABERTA "por deslize" por 5 s |
| CA12 | logado | deslizar só um pouco e soltar | o botão volta; porta trancada |
| CA13 | logado | ícone de câmera, rosto do dono | "Rosto reconhecido. Porta liberada!" e PORTA ABERTA |
| CA14 | logado | ícone de câmera, rosto de outro cadastrado | "não corresponde à sua conta"; resposta sem nome do outro |
| CA15 | logado | editar nome/apartamento e salvar | "Dados atualizados." e o cabeçalho mostra o novo nome |
| CA16 | logado | trocar o e-mail sem a senha atual | pede a senha atual; com ela, o login passa a ser o novo e-mail |
| CA17 | logado em 2 navegadores | trocar a senha em um | o outro é desconectado; a senha antiga deixa de valer |
| CA18 | logado | excluir a conta | volta ao início; login deixa de funcionar; "Nenhuma pessoa cadastrada." |
| CA19 | sem sessão | chamar qualquer `/api/me…` | 401 |
| CA20 | qualquer | inspecionar logs | sem e-mail, senha ou nome |

## Fora do escopo
Recuperação de senha, verificação de e-mail, administrador/síndico e re-escanear o rosto de uma conta
existente (para trocar o rosto: excluir a conta e cadastrar de novo). Ver ADR-005.
