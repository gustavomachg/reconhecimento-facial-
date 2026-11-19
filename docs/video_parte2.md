# Roteiro do vídeo — Parte 2 (aplicação)

Duração sugerida: **7 a 9 minutos**. Cobre tudo o que a §42 do enunciado pede: inicialização,
cadastro, reconhecimento, pessoa desconhecida, situação de erro e arquitetura.

## Regras de privacidade para a gravação (§42)
- Aparece na câmera **somente o próprio aluno**.
- Para a "pessoa desconhecida", **não** usar um familiar ou amigo. Use uma foto de domínio público
  ou um rosto gerado por IA aberto no celular.
- Antes de gravar, **esvaziar o banco** (`embeddings/facerec.db`) para o início mostrar
  "Nenhuma pessoa cadastrada.". Ao final, excluir a própria conta.
- Use um **e-mail fictício** (ex.: `demo@exemplo.com`), uma **senha só da demonstração** e um
  apartamento fictício. Nunca digite uma senha real diante da câmera.
- Não mostrar a pasta `pessoas/`, `data/private/`, `embeddings/` nem o terminal com caminhos de fotos.

## Preparação (fora da gravação)
```bash
python scripts/download_arcface.py --pack buffalo_s   # modelo vencedor (MobileFaceNet)
pytest -q                                             # deixar a saída verde pronta para mostrar
```
Ter à mão: uma imagem **sem rosto** (paisagem) e o rosto sintético/de domínio público no celular.

---

## 1. Abertura e arquitetura (≈ 1 min 30)
- Problema: **portaria de condomínio** — cada pessoa se cadastra escaneando o rosto e depois abre a
  porta pelo rosto ou por um deslize, com privacidade (LGPD). Interface própria do projeto.
- Pipeline (README):
  `Imagem → MTCNN (detecção + 5 landmarks) → alinhamento 112×112 → MobileFaceNet → embedding L2 (512) → protótipo/cosseno → threshold`.
- **Por que este modelo** (ADR-001): no LFW empata com o ArcFace iResNet-50 (EER 0,32% × 0,30%),
  supera a FaceNet (0,93%) e é ~18× mais rápido (5 ms/face): viabiliza a leitura pela câmera em CPU.
- Camadas do código: `detection` · `models` · `matching` · `auth` (senha e sessão) · `storage`
  (SQLite, UUID opaco) · `service` (casos de uso) · `app` (FastAPI + página).
- Spec-Driven: abrir `docs/specs/009-accounts-and-login.md` e mostrar o fluxo e a tabela de
  critérios de aceitação. Cada tela do vídeo corresponde a uma spec.

## 2. Inicialização (≈ 40 s)
```bash
uvicorn facerec.app.main:app
```
- Abrir `http://127.0.0.1:8000`. Destacar:
  - as duas escolhas, **Entrar** e **Cadastrar-se**;
  - **"Nenhuma pessoa cadastrada."** (estado de quem acabou de clonar, §43);
  - no rodapé: **Local · Modelo MobileFaceNet (E08) · comparador · threshold**, a configuração da Parte 1.

## 3. Cadastro em 3 etapas (≈ 1 min 30)
- **Cadastrar-se → Etapa 1 (Rosto):** a câmera liga; clicar **"Escanear rosto"** → 5 fotos em
  sequência, movendo levemente o rosto. Mostrar que dá para remover uma miniatura (×).
- **Etapa 2 (Dados):** nome, apartamento fictício, **Morador ou Visitante**, e-mail e senha.
  Digitar senhas diferentes de propósito → "As senhas não são iguais." (primeiro erro tratado).
- **Etapa 3 (Confirmar):** resumo dos dados. Clicar em concluir **sem** marcar a autorização →
  aviso. Marcar: é o consentimento do titular (§14).
- Concluir → cai direto na página da conta: **"Olá, <nome>"**, etiqueta MORADOR, apartamento.
- Comentar: só os **embeddings** são guardados (não as fotos), e a senha só como **hash**.

## 4. Abrir a porta: deslize e rosto (≈ 1 min 30)
- Mostrar **PORTA TRANCADA**. A porta é simulada, e há **duas formas** de abrir.
- **Forma 1 — deslizar:** arrastar só um pouco e soltar (não abre); arrastar até o fim →
  **PORTA ABERTA** "por deslize", com a barra de tempo, e tranca sozinha em 5 s.
- **Forma 2 — rosto:** tocar no **ícone de câmera** → a câmera abre, caixa **verde "Você"** sobre o
  rosto, "Rosto reconhecido. Porta liberada!", a porta abre. Isso é o reconhecimento por **vídeo**
  (spec 007): quadros enviados em sequência até reconhecer.
- Mostrar "Últimas aberturas" (só na tela, não é gravado).

## 5. Pessoa desconhecida (≈ 45 s)
- Tocar no ícone de câmera e mostrar o **rosto sintético** no celular → caixa **vermelha**,
  **"Pessoa não cadastrada. A porta continua trancada."**; depois de algumas tentativas aparece
  "Tentar de novo".
- Explicar o threshold (ADR-003): abaixo dele o sistema prefere não abrir. Nada é guardado sobre
  esse rosto.

## 6. Situações de erro (≈ 1 min)
- Ícone de câmera **sem ninguém no quadro** → "Nenhuma face detectada na imagem." e a leitura continua.
- Rosto muito longe → "Face muito pequena… Aproxime-se da câmera."
- **Sair** e tentar **Entrar com a senha errada** → "E-mail ou senha incorretos." (a mesma mensagem
  para e-mail inexistente). Comentar o bloqueio após 5 tentativas.
- **Cadastrar-se de novo com o mesmo rosto** (outro e-mail) → volta à etapa 1 com "Este rosto já está
  cadastrado" — sem dizer de quem (duplicidade).

## 7. Página do usuário e exclusão — Privacy by Design (≈ 1 min)
- Entrar com a senha certa → aba **Usuário**: nome, tipo e e-mail.
- **Informações pessoais:** mudar o apartamento ou o tipo e salvar → "Dados atualizados.".
  Comentar: trocar e-mail ou senha exige a senha atual.
- **Excluir minha conta** → confirmação → volta ao início com **"Nenhuma pessoa cadastrada."**.
- Tentar entrar de novo → não existe mais: a exclusão removeu cadastro, e-mail, senha e embeddings
  (`secure_delete` + `VACUUM`).

## 8. Qualidade e encerramento (≈ 45 s)
- Mostrar `pytest` passando (dados 100% sintéticos) e o CI verde no GitHub.
- Mostrar o `.gitignore`/hook de pre-commit: nenhum dado biométrico, e-mail ou senha vai ao repositório.
- Limitações: sem anti-spoofing (uma foto na tela pode ser reconhecida), uma face por quadro,
  threshold calibrado com dataset pequeno, sem recuperação de senha.
