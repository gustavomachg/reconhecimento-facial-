# Spec 005 — Interface web com câmera

**Status:** implementada · **Módulos:** `src/facerec/app/main.py`, `src/facerec/app/static/index.html`
**Testes:** `tests/test_app.py`
**Detalhes por tela:** [008 — portaria](008-condominium-gate.md) ·
[009 — contas e login](009-accounts-and-login.md) · [007 — vídeo](007-live-video-recognition.md) · [010 — movimento e 3D](010-motion-and-3d.md)

## Requisitos
1. Aplicação web **local** (FastAPI + Uvicorn em `127.0.0.1:8000`); nenhuma imagem sai da máquina.
   **Nada é pedido à internet:** sem fontes, scripts ou imagens de outros hosts. A única
   biblioteca de interface (three.js, para o fundo 3D — spec 010) fica copiada no repositório
   e é servida pelo próprio app em `/static`.
2. Telas (navegação por `#`): **Início** (Entrar / Cadastrar-se) · **Entrar** · **Cadastro em 3
   etapas** (rosto → dados → confirmar) · **Porta** (deslizar ou câmera) · **Usuário** (dados
   pessoais, senha, excluir conta).
3. Webcam via `getUserMedia`: usada no escaneamento do cadastro e em "Abrir com o rosto". A câmera
   só fica ligada enquanto a tela precisa dela e é desligada ao sair. Upload de arquivos continua
   disponível no cadastro.
4. Estado inicial após clonar: **"Nenhuma pessoa cadastrada."** na página de início.
5. Mensagens de erro em linguagem clara, sempre visíveis: aviso (laranja) para 400/409/422/429,
   erro (vermelho) para 500 e servidor fora do ar. Nenhuma ação falha em silêncio.
6. Usável no celular (uma coluna, abas fixas embaixo) e pelo teclado (o deslizante aceita Enter;
   foco visível; o modal fecha com Esc).
7. O rodapé mostra o modelo, o comparador e o threshold em uso.

## API
Tabela completa na [spec 009](009-accounts-and-login.md#api). Resumo dos códigos de erro:
entrada inválida → 400; sem sessão ou credenciais erradas → 401; e-mail em uso → 409; erros de
domínio do rosto → 422 `{code, message}`; bloqueio de login → 429; arquivo > 10 MB → 413;
erro inesperado (ex.: modelo não baixado) → 500 `{code: "internal_error", message}`.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | repositório recém-clonado | abrir a página | "Entrar" / "Cadastrar-se" e "Nenhuma pessoa cadastrada." |
| CA2 | câmera ligada no cadastro | "Escanear rosto" e concluir as etapas | conta criada; cai na página da Porta |
| CA3 | logado | ícone de câmera, rosto do dono | "Rosto reconhecido. Porta liberada!" |
| CA4 | logado | ícone de câmera, rosto não cadastrado | "Pessoa não cadastrada." |
| CA5 | logado | ícone de câmera, sem rosto no quadro | "Nenhuma face detectada na imagem." |
| CA6 | aparelho sem câmera | abrir o cadastro | "Nenhuma câmera encontrada neste aparelho." e opção de enviar fotos |
| CA7 | tela de 390 px de largura | navegar por todas as telas | sem rolagem horizontal |
