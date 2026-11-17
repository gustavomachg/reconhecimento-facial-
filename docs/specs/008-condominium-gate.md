# Spec 008 — Portaria de condomínio (caso de uso da aplicação)

**Status:** implementada · **Módulos:** `storage.py` (`unit`, `kind`), `service.py`
(`enroll`, `verify_face`, `open_door`), `app/main.py`, `app/static/index.html`
**Testes:** `tests/test_app.py`, `tests/test_service.py`, `tests/test_storage.py`
**Relacionadas:** [009 — contas e login](009-accounts-and-login.md)

## Contexto
O enunciado (§40) deixa a interface livre. Adotamos o caso de uso de **controle de acesso de
condomínio**, no estilo dos aplicativos de portaria: cada pessoa se cadastra escaneando o rosto,
informa se é **morador** ou **visitante**, e depois abre a porta de duas formas — **deslizando o
dedo** ou **pelo rosto** (ícone de câmera).

É o mesmo pipeline das specs 001–007, apresentado com a linguagem do domínio:

| Resultado do sistema | No app | Porta |
|---|---|---|
| Rosto reconhecido como o dono da conta (`score ≥ threshold`) | "Rosto reconhecido. Porta liberada!" | abre |
| Rosto de outra pessoa cadastrada | "Este rosto não corresponde à sua conta." | continua trancada |
| Pessoa desconhecida (`score < threshold`) | "Pessoa não cadastrada." | continua trancada |
| Erro de domínio (sem face, várias faces, qualidade) | aviso com o motivo; a câmera continua tentando | continua trancada |
| Deslizar até o fim | "Aberta por deslize" | abre |

O nome ("Portaria Facial") e o visual são próprios do projeto; não reproduzem a marca nem o layout
de nenhum produto comercial.

## Requisitos
1. **Cadastro por escaneamento:** o botão "Escanear rosto" tira 5 fotos em sequência (uma a cada
   0,6 s), pedindo para mover levemente o rosto. As fotos podem ser removidas uma a uma, ou o
   escaneamento refeito. Upload de arquivos continua disponível (para quem não tem câmera).
2. **Tipo da pessoa:** campo `kind` = `morador` | `visitante`, **informado por quem se cadastra**
   (o sistema não tenta adivinhar). Padrão `morador`; outro valor → 400.
3. **Apartamento/bloco:** campo opcional `unit` (até 30 caracteres). Nome até 100. Excesso → 400.
4. **Abrir pelo rosto** (`POST /api/me/door/face`): o rosto é procurado entre **todos** os
   cadastrados (identificação 1:N com o comparador e o threshold da Parte 1) e a porta só abre se
   o mais parecido for o dono da conta logada. Morador e visitante abrem do mesmo jeito.
5. **Abrir por deslize** (`POST /api/me/door/open`): disponível para quem está logado, mesmo com a
   câmera desligada.
6. **Porta simulada:** não há fechadura conectada. A página mostra "PORTA TRANCADA" / "PORTA ABERTA"
   com barra de tempo; depois de `door.open_seconds` (`configs/app.yaml`, padrão 5 s) ela tranca
   sozinha. Cada abertura gera o log de auditoria `door_open` com `method` (`face`/`manual`) e o
   UUID da pessoa — **sem nome**.
7. **Consentimento obrigatório (LGPD, §14):** o cadastro exige `consent=true`; sem isso → 400.
8. **Quem não é o dono da conta não é identificado:** a resposta nunca traz nome ou dados de outra
   pessoa, e nada é guardado sobre um rosto não cadastrado (nem foto, nem embedding).
9. **Últimas aberturas:** lista com as 8 últimas, mantida **só na memória da página** (não vai para
   disco nem banco; some ao recarregar).
10. **Logs:** nome, apartamento, e-mail e senha não aparecem em logs (`FORBIDDEN_KEYS`).
11. **Banco antigo:** as colunas novas são adicionadas por migração (`ALTER TABLE`), sem perder
    cadastros existentes.
12. Textos vindos do servidor (nome, apartamento) entram na página como **texto**, nunca como HTML.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | câmera disponível | "Escanear rosto" | 5 miniaturas do rosto aparecem, uma a uma |
| CA2 | cadastro com tipo "Visitante" | concluir | a conta mostra a etiqueta VISITANTE |
| CA3 | cadastro sem consentimento | concluir | aviso na página; a API recusa com 400 e nada é salvo |
| CA4 | morador logado | ícone de câmera, mostra o próprio rosto | "Porta liberada!" e **PORTA ABERTA** por 5 s |
| CA5 | visitante logado | ícone de câmera, mostra o próprio rosto | porta aberta do mesmo jeito |
| CA6 | logado | ícone de câmera, rosto não cadastrado | "Pessoa não cadastrada.", `door_open=false` |
| CA7 | logado | deslizar até o fim | porta aberta por 5 s, "Aberta por deslize"; log `door_open` `method=manual` |
| CA8 | porta aberta | passam `door.open_seconds` | volta a "PORTA TRANCADA" |
| CA9 | conta excluída | — | o rosto deixa de abrir a porta; pode ser cadastrado de novo |
| CA10 | `kind = "porteiro"` | cadastrar | 400 "Informe se a pessoa é morador ou visitante." |
| CA11 | banco criado antes da spec | abrir a aplicação | cadastros preservados |

## Fora do escopo
- Fechadura/catraca real, validade/horário para visitantes e registro persistente de acessos (seria
  dado pessoal adicional, com base legal própria).
- Anti-spoofing (foto de foto): limitação conhecida, descrita no README.
