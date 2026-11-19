# Privacidade e LGPD

Dados biométricos são **dados pessoais sensíveis** (LGPD, Lei 13.709/2018, art. 5º, II).
Este documento registra as medidas adotadas no trabalho.

## Base legal e consentimento
- Base legal: **consentimento específico e destacado** do titular (art. 11, I), para a finalidade
  acadêmica deste trabalho.
- Participantes: **apenas adultos**, com termo de consentimento assinado.
- Os termos ficam **fora do repositório** (`consent/` está no `.gitignore` e bloqueado pelo hook).
- O participante pode revogar o consentimento a qualquer momento: seus dados são removidos
  (na aplicação, a própria pessoa usa "Excluir minha conta" — `DELETE /api/me`; no dataset,
  remoção da pasta `data/private/raw/<ID>`).

## Minimização
- Coletar só o necessário: ~5–10 fotos por sessão, 3 sessões por pessoa, com variações de
  iluminação, pose, distância, expressão, fundo e óculos.
- Pastas identificadas por **código opaco** (`P01`, `P02`…), nunca pelo nome. Nomes de arquivo
  também não devem conter nomes (renomear fotos de celular como `IMG_xxx` → `001.jpg`).
- A aplicação **não guarda fotos** por padrão (`store_photos: false`); guarda só embeddings.
- O cadastro na aplicação exige a confirmação de que a pessoa (morador ou visitante) consentiu (`consent`); o **termo
  assinado** fica com o grupo, fora do sistema e do GitHub. Nome e apartamento também são dados
  pessoais: não aparecem em logs. Quem não está cadastrado não é identificado nem registrado. A abertura da porta gera só
  um log de auditoria com a forma (rosto/botão) e o UUID, sem nome.
- **Contas (spec 009, ADR-005):** cada pessoa se cadastra e consente no próprio cadastro. O e-mail
  é dado pessoal; a senha é guardada só como **hash** (PBKDF2). Ficam no mesmo banco local dos
  embeddings (`embeddings/facerec.db`, ignorado pelo git) e são apagados junto com a conta.
  Cada pessoa só acessa os próprios dados: não há rota que liste pessoas, e a porta pelo rosto
  nunca revela de quem é um rosto diferente do dono da conta.
- No reconhecimento (foto ou contínuo pela câmera), cada quadro só existe na memória durante a
  requisição: nada é gravado em disco nem em log (só UUID, score e latência).

## Segurança
- Todo processamento é **local**; nenhuma imagem/embedding é enviado a serviços externos.
  A aplicação escuta apenas em `127.0.0.1` por padrão. Para usar a câmera de outro computador da
  **mesma rede doméstica**, ela pode ser exposta na rede local só com **HTTPS** (imagens
  criptografadas no caminho) e só durante o uso — ver "Acessar de outro computador" no README.
  Nunca expor em rede pública (faculdade, café): o cadastro é aberto a quem alcançar o endereço.
- `.gitignore` + hook de pre-commit `scripts/check_private_files.py` bloqueiam fotos, vídeos,
  bancos, embeddings, checkpoints e termos.
- Logs estruturados com **UUID opaco**; `log_event` recusa campos como `name`, `image`, `embedding`.
- Exclusão com `secure_delete` + `VACUUM` no SQLite (spec 006).
- Agentes de IA usados no desenvolvimento trabalham só com código, specs e dados sintéticos
  (enunciado §23); testes usam apenas dados sintéticos.

## Resultados
- Publicados somente **agregados** (acurácia, EER, FAR/FRR globais, contagens). Nada por participante.
- Gráficos sem rostos; notebooks sem outputs com imagens privadas.

## Retenção e eliminação
- Os dados privados são mantidos **somente até a avaliação final da disciplina**.
- Eliminação: `python scripts/purge_private_data.py --yes` (apaga `data/private/`, `pessoas/`,
  `faces/`, `embeddings/`, `checkpoints/private/`, bancos `.db`). Termos de consentimento só com
  `--include-consent`, após o prazo de guarda definido com o professor.
- Atenção a **cópias automáticas**: este projeto está numa pasta sincronizada pelo OneDrive, que
  envia os arquivos para a nuvem — um serviço externo (enunciado §22). **Recomendado:** mover o
  projeto (ou ao menos `data/private/`, `pessoas/`, `embeddings/`, `checkpoints/`) para uma pasta
  local não sincronizada, ex.: `C:/projetos/reconhecedor-facial`. Na eliminação, esvaziar também
  a lixeira local e a do OneDrive.

## Proibições
Scraping de redes sociais ou busca automática de pessoas na internet **não** são implementados.
Datasets públicos não são redistribuídos (só scripts/instruções de download da fonte oficial).
