# Spec 007 — Reconhecimento pela câmera em vídeo

**Status:** implementada · **Módulos:** `service.py` (`verify_face` → `box`), `app/main.py`
(`POST /api/me/door/face`, `GET /api/status`), `app/static/index.html` (modal "Abrir com o rosto")
**Testes:** `tests/test_app.py`, `tests/test_service.py`

## Problema
O enunciado (§38) pede que a aplicação reconheça a partir de **imagem ou vídeo**, e a §40 mostra o
resultado sobre a imagem da câmera.

## Decisão de design
O vídeo é tratado como uma **sequência de quadros** enviados à rota `POST /api/me/door/face`.
Não há streaming nem WebSocket: o pipeline (detecção → embedding → matcher → threshold) é o da
spec 003, e cada quadro é independente.
- Ao tocar no ícone de câmera, a página envia um quadro a cada ~0,7 s, **esperando a resposta do
  anterior**: nunca há requisições acumuladas, e a taxa se adapta à velocidade da máquina.
- A leitura para no primeiro quadro reconhecido, ou depois de 10 tentativas (≈ 7 s), ou quando o
  usuário cancela. Ao fechar, a câmera é desligada.
- Viável em CPU porque o modelo escolhido (MobileFaceNet, ADR-001) gasta ~5 ms/face; o custo
  dominante é a detecção MTCNN.
- Nenhum quadro é gravado: cada imagem só existe na memória durante a requisição (LGPD).

## Requisitos
1. O ícone de câmera da página principal abre a câmera e inicia a leitura contínua.
2. A resposta de `/api/me/door/face` inclui `box = [x1, y1, x2, y2]` (pixels da imagem enviada)
   quando uma face foi detectada.
3. A interface desenha a caixa sobre o vídeo: **verde** com "Você" se reconhecido, **vermelha**
   com "Não reconhecido" caso contrário; os cantos de mira mudam de cor junto.
4. Erros de domínio por quadro (422: sem face, várias faces, qualidade) aparecem como aviso e
   **não** interrompem a leitura; erros graves (500, servidor fora) interrompem.
5. Se nenhuma tentativa reconhecer: "Não foi possível reconhecer o seu rosto. A porta continua
   trancada." e o botão "Tentar de novo".
6. `GET /api/status` informa o modelo, o comparador e o threshold em uso (sem dados pessoais),
   mostrados no rodapé. Assim a demonstração deixa claro que a aplicação usa a configuração
   vencedora da Parte 1.
7. Sem câmera, sem permissão ou fora de HTTPS/localhost: mensagem clara em português.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | logado, câmera disponível | toca no ícone e mostra o rosto | caixa verde "Você", "Porta liberada!", o modal fecha e a porta abre |
| CA2 | logado | mostra um rosto não cadastrado | caixa vermelha e "Pessoa não cadastrada."; após 10 tentativas, "Tentar de novo" |
| CA3 | logado | ninguém no quadro | aviso "Nenhuma face detectada na imagem." e a leitura continua |
| CA4 | logado | `POST /api/me/door/face` com rosto | resposta tem `box` com 4 números |
| CA5 | logado | cancela durante a leitura | o modal fecha, a câmera desliga e nenhuma requisição nova é enviada |
| CA6 | aplicação iniciada | `GET /api/status` | `{model, matcher, threshold, door_seconds, n_persons, message}` sem nomes |
| CA7 | erro inesperado no servidor | qualquer rota | 500 com JSON `{code: "internal_error", message}` mostrado na página |

## Fora do escopo
- Várias faces no mesmo quadro: continua retornando `multiple_faces` (spec 001), como no cadastro.
- Anti-spoofing (foto de foto): limitação conhecida, descrita no README.
