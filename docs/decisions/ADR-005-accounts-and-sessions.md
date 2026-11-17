# ADR-005 — Contas, senha e sessão

**Status:** aceito
**Data:** 2026-11-16

## Contexto
A aplicação virou um app de portaria em que **cada pessoa se cadastra sozinha** (rosto + dados) e
depois entra com e-mail e senha para abrir a porta e editar os próprios dados (spec 009). Isso
acrescenta dados pessoais (e-mail) e um segredo (senha) a um sistema que já guarda biometria, e
exige decidir como autenticar sem depender de serviço externo (§23 do enunciado: processar
localmente).

## Decisões
1. **Uma tabela só.** A conta fica na própria linha de `persons` (`email` único, `password_hash`).
   Excluir a pessoa apaga conta, embeddings e fotos de uma vez (cascade + `secure_delete`), o que
   mantém a exclusão completa e verificável (spec 006).
2. **Senha: PBKDF2-HMAC-SHA256, 600 000 iterações, sal aleatório de 16 bytes** (`hashlib`, biblioteca
   padrão). Só o hash é guardado, no formato `pbkdf2_sha256$iterações$sal$hash`: as iterações ficam
   no próprio hash, então podem aumentar sem invalidar senhas antigas. Comparação em tempo constante.
   - *Alternativas:* bcrypt/argon2 são melhores contra GPU, mas exigem dependência compilada a mais;
     para um sistema local e acadêmico, PBKDF2 com o número de iterações recomendado pela OWASP é
     suficiente e mantém o ambiente reproduzível.
3. **Sessão: token aleatório (256 bits) guardado na memória do servidor**, entregue em cookie
   `HttpOnly` + `SameSite=Strict` (+ `Secure` quando em HTTPS), válido por 8 h.
   - `HttpOnly`: o JavaScript da página não lê o token. `SameSite=Strict`: outro site não consegue
     disparar ações com a sessão do usuário (CSRF).
   - Em memória: reiniciar o servidor desconecta todos. É aceitável aqui e evita guardar tokens em
     disco. *Alternativa descartada:* JWT — não dá para revogar sem lista de bloqueio, e trocar a
     senha/excluir a conta precisa encerrar sessões na hora.
4. **Força bruta:** 5 erros seguidos no mesmo e-mail bloqueiam novas tentativas por 60 s.
5. **Não revelar quem tem conta:** e-mail inexistente e senha errada devolvem a mesma mensagem, e o
   servidor gasta o mesmo tempo nos dois casos (calcula um hash de mentira).
   Exceção assumida: o cadastro avisa "e-mail já cadastrado" (necessário para a pessoa entender o
   erro).
6. **Cada pessoa só acessa os próprios dados.** Não existe rota que liste pessoas ou devolva dados
   de terceiros. A porta pelo rosto compara com todos (1:N), mas só responde "é você" ou "não é".
7. **Trocar e-mail ou senha exige a senha atual;** trocar a senha encerra as outras sessões.
8. **Logs sem credenciais:** `email`, `password` e `password_hash` estão em `FORBIDDEN_KEYS`;
   os eventos (`login_ok`, `login_failed`, `door_open`…) levam só o UUID.

## Consequências e limites
- O banco (`embeddings/facerec.db`, fora do git) passa a conter e-mails e hashes de senha: continua
  sendo **dado privado** e entra na eliminação final (`scripts/purge_private_data.py`).
- Não há "esqueci minha senha" (exigiria enviar e-mail por serviço externo). A saída é excluir o
  banco local ou a linha da pessoa.
- Não há verificação de e-mail nem papel de administrador/síndico.
- Sem HTTPS, senha e imagens trafegam em claro na rede: fora do `localhost`, usar o modo HTTPS do
  README.
- Cadastros antigos ou importados (`scripts/import_people.py`) não têm conta: não entram no app,
  mas continuam ocupando o rosto (um novo cadastro do mesmo rosto dá "já cadastrado").
