# Privacidade e LGPD

Dados biométricos são **dados pessoais sensíveis** (LGPD, Lei 13.709/2018, art. 5º, II).
Este documento registra as medidas adotadas no trabalho.

## Base legal e consentimento
- Base legal: **consentimento específico e destacado** do titular (art. 11, I), para a finalidade
  acadêmica deste trabalho.
- Participantes: **apenas adultos**, com termo de consentimento assinado.
- Os termos ficam **fora do repositório** (`consent/` está no `.gitignore` e bloqueado pelo hook).
- O participante pode revogar o consentimento a qualquer momento: seus dados são removidos
  (`DELETE /api/persons/{id}` na aplicação e remoção da pasta `data/private/raw/<ID>`).

## Minimização
- Coletar só o necessário: ~5–10 fotos por sessão, 3 sessões por pessoa, com variações de
  iluminação, pose, distância, expressão, fundo e óculos.
- Pastas identificadas por **código opaco** (`P01`, `P02`…), nunca pelo nome. Nomes de arquivo
  também não devem conter nomes (renomear fotos de celular como `IMG_xxx` → `001.jpg`).
- A aplicação **não guarda fotos** por padrão (`store_photos: false`); guarda só embeddings.

## Segurança
- Todo processamento é **local**; nenhuma imagem/embedding é enviado a serviços externos.
  A aplicação escuta apenas em `127.0.0.1`.
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
