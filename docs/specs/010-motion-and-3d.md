# Spec 010 — Identidade visual e porta 3D (three.js)

**Status:** implementada · **Módulos:** `app/static/index.html`, `app/static/scene.js`,
`app/static/vendor/` (three.js e fonte), `app/main.py` (`/static`)
**Testes:** `tests/test_app.py` (`test_scene_and_three_are_served_locally`,
`test_page_loads_the_scene_as_optional_module`, `test_home_door_frames_the_buttons`,
`test_font_is_served_locally`, `test_static_files_are_revalidated`,
`test_interface_requests_nothing_from_other_hosts`)

## Ideia
Uma página de abertura de estilo **editorial**: uma **palavra gigante** ("PORTARIA", em fonte
serifada) **inteira na largura da tela**, uma **porta grande só de contorno** no meio, com fresta de luz e
duas luzes fortes, um **anel de luz no chão**, e os botões como **adesivos amarelos espelhados** no
pé da porta. Antes, uma **tela de carregamento** com a porcentagem. Ao clicar, a porta abre.

A direção veio de uma referência dada pelo grupo (página de 3D com tipografia gigante, porta em
linhas finas e selo amarelo). Foram aproveitados a *composição e o clima*; o desenho, as cores e o
código são do projeto.

| Momento | O que se vê |
|---|---|
| Primeira visita da aba | Tela escura de carregamento: "Abrindo a portaria — aguarde", porcentagem grande, linha de progresso e legendas em fonte mono. A porcentagem segue o que de fato carrega (configuração, fontes, cena 3D). |
| Início | Topo com a marca **PORTARIA.**, "Entrar" e a pílula branca "Cadastrar-se". Palavra gigante inteira; porta grande de linhas finas em "V" no meio, com fresta de luz e duas luzes fortes; anel de luz e névoa no chão; adesivos **● ENTRAR** e **CADASTRAR-SE ●** no pé da porta; texto em caixa alta e o estado ("● Nenhuma pessoa cadastrada.") em fonte mono. |
| Clique em Entrar/Cadastrar-se | Os adesivos somem, a porta abre, sai luz e **uns poucos pontos** pela abertura, a câmera atravessa; só então a tela troca. |
| Conta | A porta aparece grande **atrás dos cartões**; o estado vira um selo: amarelo "● PORTA TRANCADA", verde "● PORTA ABERTA". Ao abrir, a porta abre e solta alguns pontos. |
| Erro ou acesso negado | A porta treme um instante e as luzes ficam vermelhas. |

## Decisões
- **Sem brilho exagerado nem degradê:** botões sólidos, sombras duras (deslocadas) em vez de neon,
  cartões lisos. O único brilho está na fresta e nas duas luzes da porta, como na referência.
- **Paleta:** azul-royal que desce até quase preto, off-white, amarelo (`#ffd166`) para ações e
  selos. Verde e vermelho só significam "ok" e "erro".
- **Tipografia:** Playfair Display (licença SIL OFL, copiada em `static/vendor/` com o texto da
  licença) para a palavra gigante e os títulos; fontes do sistema no resto.
- **three.js r160 (MIT) copiado para o repositório** e servido pelo próprio app. **Nada é carregado
  de CDN ou de outro host:** a promessa "100% local" continua valendo e o app funciona sem internet.
- **A cena é decoração, não função.** `scene.js` é um módulo separado e opcional: se o WebGL não
  existir ou falhar, a porta de CSS (mesmo desenho, em linhas) faz o mesmo papel e a troca de tela
  é imediata.
- **Camadas:** névoa e palavra (CSS) < cena 3D < adesivos/texto (CSS). Assim as linhas e as luzes
  da porta passam por cima das letras e os botões ficam sempre legíveis e clicáveis.
- **Página → cena por eventos** (`portaria:view`, `enter`, `scan`, `door`, `flash`), disparados por
  `fx()` em `index.html`. A cena não conhece a API nem recebe dado pessoal: só o nome da tela e
  verdadeiro/falso.
- **A porta do início é ancorada a um bloco do layout** (`#doorAnchor`), então acompanha o layout em
  qualquer tamanho de tela. O bloco tem a mesma proporção da porta (2,2 : 2,6) para não distorcer.

## Requisitos
1. Início com palavra gigante, porta de contorno e dois adesivos amarelos espelhados no pé da porta
   (mesmo estilo; o ponto fica do lado de fora de cada um). Os links do topo levam às mesmas telas.
2. **Entrada da página:** na primeira visita da aba, a tela de carregamento (≥ 1,3 s, ≤ 4 s, mesmo
   se algo falhar) sobe como uma cortina; depois as letras da palavra sobem uma a uma, a porta
   aparece já se fechando e entram os adesivos, o texto e o estado. Nas visitas seguintes da mesma
   aba, ou com "reduzir movimento", não há tela de carregamento.
3. **Profundidade:** folhas com espessura (arestas em perspectiva) formando um "V" leve, anel de
   luz e névoa no chão, e paralaxe leve (a câmera e a palavra se movem em sentidos opostos com o
   ponteiro).
4. **Hover:** passar o mouse (ou focar pelo teclado) em um adesivo entreabre a folha do lado dele,
   deixando escapar um pouco de luz.
5. **Travessia (≈ 0,9 s):** ao clicar, os adesivos somem, a porta abre, saem luz e poucos pontos
   pequenos, e a rota muda. Um segundo clique durante a travessia é ignorado.
6. **Pontos só na abertura** (≈ 110, bem pequenos, sumindo em ~1,3 s). Nunca ficam sobre o texto
   em repouso.
7. **Conta:** a porta acompanha o estado simulado — aberta enquanto a porta do condomínio está
   aberta, fechada quando tranca. O estado também aparece como selo na página.
8. **Erro:** tremida curta da porta e luzes vermelhas.
9. **Transições de interface:** telas, etapas do cadastro, cartões, modal e avisos entram com
   animação curta (≤ 0,5 s).
10. **Acessibilidade:** com "reduzir movimento" ativado no sistema, a cena fica parada (um quadro
   por mudança), não há pontos nem travessia e as animações de interface são desligadas. O título
   da página existe como `h1` para leitores de tela; a palavra gigante é decorativa (`aria-hidden`).
11. **Desempenho:** resolução limitada (pixel ratio ≤ 2), poucos objetos, e a animação para quando
   a aba está em segundo plano.
12. A página e os arquivos de `/static` são servidos com `Cache-Control: no-cache` (o navegador
   confere se há versão nova), para que uma atualização apareça sem limpar o cache.

## Layout responsivo do início
Tudo no início deriva do **tamanho da palavra gigante** (`--fs`, em `index.html`): a largura da
porta (`--door-w`) e o espaço do texto sob o anel do chão vêm dela. Assim a composição mantém a
mesma proporção em qualquer tela.
- `--fs` é limitada pela **largura** e pela **altura** (`min(19,5vw, 40vh)`): a palavra aparece
  **inteira** (ocupa ~91% da largura). Diferente da referência, ela não é cortada nas bordas.
- `--door-w` = `clamp(220px, min(--fs × 1,9, 42vh), 520px)` (em celular `min(62vw, 260px)`): a
  porta é grande, como na referência, e a palavra atravessa a porta na altura das luzes.
- A faixa da palavra ocupa a largura da tela com `overflow-x: clip`, para a névoa larga do chão
  não criar rolagem lateral.

### A porta (como na referência)
- Duas folhas só de contorno (lilás), formando um "V": o meio fica mais longe (`FOLD`), e as folhas
  se alongam para se encontrarem na fresta.
- Fresta ciano de cima a baixo, com brilho.
- Duas luzes: miolo branco de borda nítida e halo azul, uma de cada lado da fresta.
- Anel de luz turquesa no chão, com ~2× a largura da porta, achatado, centrado no pé da porta.
- A tela inicial é **centralizada na vertical** e cabe sem rolagem de 375 a 1920 px de largura
  (inclusive 1366×650).
- O texto de apresentação fica abaixo do anel e da névoa do chão, nunca por cima deles.

## Critérios de aceitação
| # | Dado | Quando | Então |
|---|------|--------|-------|
| CA1 | navegador com WebGL | abrir o site | palavra gigante inteira, porta de linhas em "V" com fresta ciano e duas luzes, anel no chão e dois adesivos espelhados |
| CA2 | tela inicial | "Entrar" ou "Cadastrar-se" (adesivo ou topo) | porta abre, saem poucos pontos, a câmera atravessa e a nova tela aparece |
| CA3 | logado | deslizar para abrir | selo verde "PORTA ABERTA" e a porta ao fundo abre |
| CA4 | logado | rosto não reconhecido | a porta treme e as luzes ficam vermelhas |
| CA5 | sistema com "reduzir movimento" | usar o app | cena parada, sem travessia nem pontos; tudo funciona |
| CA6 | navegador sem WebGL | abrir o site | porta em linhas de CSS e app funcionando |
| CA7 | computador sem internet | abrir o site | interface completa, inclusive a fonte (nenhum recurso externo) |
| CA8 | qualquer | inspecionar `index.html` e `scene.js` | nenhuma URL de outro host |
| CA9 | tela de 390 px | navegar por todas as telas | sem rolagem horizontal; botões do início legíveis |
| CA10 | notebook de 1366×650, 1280×720, monitor de 1920×1080 e celular | abrir o início | adesivos lado a lado no pé da porta; texto de apresentação legível; sem rolagem lateral nem vertical |
| CA11 | primeira visita da aba | abrir o site | tela de carregamento conta até 100% e sobe; recarregar na mesma aba não mostra de novo |

## Fora do escopo
Modelos 3D de rosto, avatares ou qualquer uso da imagem da pessoa na cena.
