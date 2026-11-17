// Porta da Portaria Facial (three.js, servido localmente — nada vem de CDN).
//
// Estilo: porta só de CONTORNO, em linhas finas e painéis quase transparentes, com uma fresta de
// luz no meio e duas luzes pequenas ("olhos") onde ficariam os puxadores, como um esboço de
// arquitetura. Sem madeira, metal ou iluminação realista: só linha, transparência e um pouco de luz.
//
//   - Início: a porta fica na frente de uma palavra gigante ("PORTARIA", em HTML), sobre um anel
//     de luz no chão; os botões são adesivos amarelos no pé dela (passar o mouse entreabre a folha
//     do lado do botão). Ao escolher Entrar/Cadastrar-se, as folhas abrem, sai luz e uns poucos
//     pontos pela abertura, e a câmera atravessa a porta.
//   - Conta: a porta aparece grande atrás dos cartões; abre quando o acesso é liberado e fecha
//     quando tranca. As luzes trocam de cor conforme o estado.
//   - Erro: a porta treme um instante e as luzes ficam vermelhas.
//
// É só decoração: a página funciona igual sem ela. Se o WebGL não existir, o canvas é removido e a
// página usa as linhas da porta em CSS.
// Eventos recebidos da página (ver `fx()` em index.html):
//   portaria:view  {name}          troca de tela
//   portaria:enter {}              clicou em Entrar/Cadastrar-se no início -> abre e atravessa
//   portaria:scan  {active}        escaneando o rosto (luzes amarelas)
//   portaria:door  {open}          porta do condomínio abriu/fechou
//   portaria:flash {kind: ok|bad}  resultado de uma ação
// Com "reduzir movimento" ativado no sistema, a cena fica parada (um quadro por mudança).

import * as THREE from "./vendor/three.module.min.js";

const W = 2.2, H = 2.6; // vão da porta (as duas folhas), em unidades locais — mesma proporção do CSS (2,2 : 2,6)
const OPEN_ANGLE = 1.65; // ~95 graus
const FOLD = 0.3; // as folhas fechadas formam um "V" bem visível: o meio fica mais longe, como na referência
const AJAR = 0.34; // quanto a folha entreabre quando o mouse passa no botão dela
const LINE = 0x7c84ff, SEAM = 0x3ee8ff, PANEL = 0x3b4cff, WARM = 0xffd166, RING = 0x30d8bf;
const EYE = { idle: 0x5fbaff, scan: 0xffd166, ok: 0x6ef0a6, bad: 0xff6b5e }; // luz dos "olhos"
// porta nas telas de dentro: grande e centralizada ATRÁS dos cartões
const DOOR_VIEWS = { door: { pos: [0, 0.2, -5], size: 2.5, opacity: 0.45 } };

const canvas = document.getElementById("bg3d");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
} catch (e) {
  canvas.remove(); // sem WebGL: a página segue normal, com as linhas da porta em CSS
}

if (renderer) start();

function start() {
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 9);

  // textura de luz suave (gerada aqui, sem arquivo de imagem)
  function glowTexture() {
    const c = document.createElement("canvas"); c.width = c.height = 64;
    const g = c.getContext("2d"), grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.28, "rgba(255,255,255,.72)");
    grad.addColorStop(0.6, "rgba(255,255,255,.16)"); grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }
  const glowTex = glowTexture();
  // miolo das luzes: disco branco de borda nítida, com uma franja curta (a "esfera" da referência)
  function orbTexture() {
    const c = document.createElement("canvas"); c.width = c.height = 64;
    const g = c.getContext("2d"), grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.5, "rgba(255,255,255,1)");
    grad.addColorStop(0.68, "rgba(235,250,255,.45)"); grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }

  // ------------------------------------------------------------ peças
  const door = new THREE.Group();
  scene.add(door);
  const fade = []; // materiais que acompanham a opacidade da porta: {m, base}
  const mat = (m, base = 1) => { m.transparent = true; m.toneMapped = false; fade.push({ m, base }); return m; };
  const lineMat = mat(new THREE.LineBasicMaterial({ color: LINE }), 0.9);
  // painel quase invisível: a palavra gigante atrás continua branca, como na inspiração
  const panelMat = mat(new THREE.MeshBasicMaterial({ color: PANEL, side: THREE.DoubleSide, depthWrite: false }), 0.07);
  const portalMat = mat(new THREE.MeshBasicMaterial({ color: 0xe9fbff, depthWrite: false }), 0); // luz que aparece ao abrir
  const seamMat = mat(new THREE.MeshBasicMaterial({ color: SEAM, blending: THREE.AdditiveBlending, depthWrite: false }), 1);
  const seamGlowMat = mat(new THREE.SpriteMaterial({ map: glowTex, color: SEAM, blending: THREE.AdditiveBlending, depthWrite: false }), 0.5);
  const eyeMat = mat(new THREE.SpriteMaterial({ map: glowTex, color: EYE.idle, blending: THREE.AdditiveBlending, depthWrite: false }), 1);
  const coreMat = mat(new THREE.SpriteMaterial({ map: orbTexture(), color: 0xffffff, blending: THREE.AdditiveBlending, depthWrite: false }), 1);

  // o que aparece quando as folhas abrem: uma área clara (portal) com uma faixa de piso
  const portal = new THREE.Mesh(new THREE.PlaneGeometry(W, H), portalMat); portal.position.z = -0.3; door.add(portal);

  // fresta central e brilho vertical
  const seam = new THREE.Mesh(new THREE.PlaneGeometry(0.035, H), seamMat); seam.position.z = 0.03; door.add(seam);
  const seamGlow = new THREE.Sprite(seamGlowMat); seamGlow.scale.set(0.7, H * 1.15, 1); seamGlow.position.z = 0.02; door.add(seamGlow);

  // folhas: painel quase invisível + contorno com espessura (as arestas aparecem quando a folha gira)
  function makeLeaf(side) { // side: -1 esquerda, +1 direita; a dobradiça fica na borda externa
    const dir = -side; // sentido da dobradiça até o centro
    const pivot = new THREE.Group(); pivot.position.x = side * (W / 2); door.add(pivot);
    const body = new THREE.Group(); body.position.x = dir * (W / 4); pivot.add(body);
    body.add(new THREE.Mesh(new THREE.PlaneGeometry(W / 2, H), panelMat));
    body.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(W / 2, H, 0.03)), lineMat));
    pivot.userData = { body, dir };
    return pivot;
  }
  const leafL = makeLeaf(-1), leafR = makeLeaf(1);

  // névoa de luz no chão, sob a porta
  const hazeMat = mat(new THREE.SpriteMaterial({ map: glowTex, color: 0x3a4cff, blending: THREE.AdditiveBlending, depthWrite: false }), 0.55);
  const haze = new THREE.Sprite(hazeMat); haze.scale.set(W * 3.6, 1.1, 1); haze.position.set(0, -H / 2 - 0.04, 0.3); door.add(haze);

  // as duas luzes fortes, onde ficariam os puxadores
  const eyes = [];
  for (const sx of [-1, 1]) {
    const halo = new THREE.Sprite(eyeMat); halo.scale.set(1.5, 1.5, 1); halo.position.set(sx * 0.55, 0.02, 0.08); door.add(halo);
    const core = new THREE.Sprite(coreMat); core.scale.set(0.5, 0.5, 1); core.position.copy(halo.position); door.add(core);
    eyes.push(halo, core);
  }

  // anel de luz no chão (desenhado num canvas com desfoque: brilha sem precisar de pós-processamento)
  function ringTexture() {
    const c = document.createElement("canvas"); c.width = 1024; c.height = 256;
    const g = c.getContext("2d");
    g.shadowColor = "rgba(63,230,210,1)"; g.lineWidth = 5;
    for (const [blur, alpha] of [[26, 0.55], [10, 0.9], [0, 1]]) {
      g.shadowBlur = blur; g.strokeStyle = `rgba(150,255,240,${alpha})`;
      g.beginPath(); g.ellipse(512, 128, 470, 60, 0, 0, Math.PI * 2); g.stroke();
    }
    return new THREE.CanvasTexture(c);
  }
  const ringMat = mat(new THREE.SpriteMaterial({ map: ringTexture(), color: RING, blending: THREE.AdditiveBlending, depthWrite: false }), 0.55);
  const ring = new THREE.Sprite(ringMat); ring.scale.set(W * 2.3, W * 2.3 / 4, 1); ring.position.set(0, -H / 2 - 0.02, 0.4); door.add(ring);

  // ------------------------------------------------------------ pontos (só na abertura)
  const N = 110;
  const pos = new Float32Array(N * 3), vel = new Float32Array(N * 3), col = new Float32Array(N * 3);
  const cA = new THREE.Color(0xffffff), cB = new THREE.Color(WARM), cC = new THREE.Color(SEAM), tmp = new THREE.Color();
  const rnd = (a, b) => a + Math.random() * (b - a);
  for (let i = 0; i < N; i++) { tmp.copy(Math.random() < 0.5 ? cA : Math.random() < 0.5 ? cB : cC); col.set([tmp.r, tmp.g, tmp.b], i * 3); }
  const ptsGeo = new THREE.BufferGeometry();
  ptsGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  ptsGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const dot = (() => { // círculo de borda dura
    const c = document.createElement("canvas"); c.width = c.height = 32;
    const g = c.getContext("2d"); g.fillStyle = "#fff"; g.beginPath(); g.arc(16, 16, 14, 0, Math.PI * 2); g.fill();
    return new THREE.CanvasTexture(c);
  })();
  const ptsMat = new THREE.PointsMaterial({ size: 0.045, map: dot, vertexColors: true, transparent: true, opacity: 0, depthWrite: false, toneMapped: false });
  const points = new THREE.Points(ptsGeo, ptsMat); points.frustumCulled = false; points.visible = false; scene.add(points);
  let spark = 0; // 1 = acabou de sair; vai a 0 em ~1,3 s

  // ------------------------------------------------------------ estado
  const state = { view: "home", scanning: false, doorOpen: false, homeOpen: false, flash: null, flashUntil: 0, shake: 0, hover: 0, pointer: new THREE.Vector2(0, 0) };
  const cur = { pos: new THREE.Vector3(), s: 1, opacity: 0, open: 0, camZ: 9, eye: new THREE.Color(EYE.idle), ajarL: 0, ajarR: 0 };
  const target = { pos: new THREE.Vector3(), eye: new THREE.Color() };
  const anchorEl = document.getElementById("doorAnchor");

  function mood() {
    if (state.flash && performance.now() < state.flashUntil) return state.flash;
    if (state.doorOpen) return "ok";
    if (state.scanning) return "scan";
    return "idle";
  }
  const halfHeight = () => Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 9; // no plano z = 0
  const portrait = () => camera.aspect < 0.9;

  // No início a porta é ancorada ao bloco dos botões (#doorAnchor): acompanha o layout em
  // qualquer tela e também a rolagem.
  function doorTarget() {
    if (state.view === "home" && anchorEl) {
      const r = anchorEl.getBoundingClientRect();
      if (r.width && r.height) {
        const hh = halfHeight(), hw = hh * camera.aspect, w = window.innerWidth, h = window.innerHeight;
        target.pos.set(((r.left + r.width / 2) / w * 2 - 1) * hw, -((r.top + r.height / 2) / h * 2 - 1) * hh, 0);
        return { s: (r.width * ((2 * hh) / h)) / W, opacity: 1, open: state.homeOpen ? 1 : 0 };
      }
    }
    const v = DOOR_VIEWS[state.view];
    if (!v) return { s: cur.s, opacity: 0, open: 1 }; // já atravessada: some aberta
    target.pos.set(v.pos[0], portrait() ? v.pos[1] + 0.8 : v.pos[1], v.pos[2]);
    return { s: portrait() ? v.size * 0.78 : v.size, opacity: portrait() ? v.opacity * 0.8 : v.opacity, open: state.doorOpen ? 1 : 0 };
  }

  // poucos pontos saem pela abertura da porta em direção à câmera
  function burst() {
    const hw = (W / 2) * cur.s * 0.7, hh = (H / 2) * cur.s * 0.55;
    for (let i = 0; i < N; i++) {
      pos[i * 3] = cur.pos.x + rnd(-hw, hw); pos[i * 3 + 1] = cur.pos.y + rnd(-hh, hh); pos[i * 3 + 2] = cur.pos.z - 0.1;
      vel[i * 3] = rnd(-1.6, 1.6); vel[i * 3 + 1] = rnd(-1.1, 1.4); vel[i * 3 + 2] = rnd(1.8, 5.5);
    }
    spark = 1; points.visible = true;
  }

  // ------------------------------------------------------------ quadro
  let last = performance.now(), time = 0, raf = 0;

  function step(dt, instant) {
    time += dt;
    const t = doorTarget();
    target.eye.setHex(EYE[mood()]);
    const ease = (rate) => (instant ? 1 : 1 - Math.exp(-dt * rate)); // suavização independente do FPS
    const k = ease(3.4), home = state.view === "home";
    cur.pos.lerp(target.pos, home ? ease(14) : k); // no início gruda no layout
    cur.s += (t.s - cur.s) * (home ? ease(14) : k);
    cur.opacity += (t.opacity - cur.opacity) * ease(4);
    cur.open += (t.open - cur.open) * ease(3.4);
    cur.eye.lerp(target.eye, ease(10));
    cur.camZ += ((state.homeOpen ? 6.4 : 9) - cur.camZ) * ease(2.6); // atravessando a porta
    state.shake = Math.max(0, state.shake - dt * 2.2);

    door.visible = cur.opacity > 0.01;
    door.position.copy(cur.pos);
    door.position.x += Math.sin(time * 55) * 0.05 * state.shake * cur.s; // tremida de "negado"
    door.scale.setScalar(cur.s);
    cur.ajarL += ((state.hover === -1 ? 1 : 0) - cur.ajarL) * ease(7);
    cur.ajarR += ((state.hover === 1 ? 1 : 0) - cur.ajarR) * ease(7);
    const closedL = FOLD + cur.ajarL * AJAR, closedR = FOLD + cur.ajarR * AJAR;
    leafL.rotation.y = closedL * (1 - cur.open) + cur.open * OPEN_ANGLE;
    leafR.rotation.y = -(closedR * (1 - cur.open) + cur.open * OPEN_ANGLE);
    // com o "V", a folha girada não alcançaria o meio: alonga um pouco para as duas se encontrarem na fresta
    const stretch = 1 + (1 / Math.cos(FOLD) - 1) * (1 - cur.open);
    for (const leaf of [leafL, leafR]) {
      const { body, dir } = leaf.userData;
      body.scale.x = stretch; body.position.x = dir * (W / 4) * stretch;
    }

    const pulse = 0.82 + Math.sin(time * 2.4) * 0.18;
    for (const f of fade) f.m.opacity = f.base * cur.opacity;
    portalMat.opacity = Math.max(cur.open, Math.max(cur.ajarL, cur.ajarR) * 0.35) * 0.88 * cur.opacity;
    seamMat.opacity = (1 - cur.open * 0.85) * cur.opacity;
    seamGlowMat.opacity = (0.55 - cur.open * 0.3) * pulse * cur.opacity;
    eyeMat.color.copy(cur.eye); eyeMat.opacity = (0.85 + cur.open * 0.15) * pulse * cur.opacity;
    for (const e of eyes) e.visible = cur.open < 0.7; // somem quando as folhas se afastam

    if (spark > 0 && !instant) {
      spark = Math.max(0, spark - dt / 1.3);
      for (let i = 0; i < N; i++) { pos[i * 3] += vel[i * 3] * dt; pos[i * 3 + 1] += vel[i * 3 + 1] * dt; pos[i * 3 + 2] += vel[i * 3 + 2] * dt; }
      ptsGeo.attributes.position.needsUpdate = true;
      ptsMat.opacity = Math.min(1, spark * 1.6) * 0.9;
      if (spark === 0) points.visible = false;
    }

    camera.position.x += (state.pointer.x * 0.6 - camera.position.x) * k; // paralaxe com o ponteiro: mostra a espessura
    camera.position.y += (state.pointer.y * 0.35 - camera.position.y) * k;
    camera.position.z = cur.camZ;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  function loop(now) { const dt = Math.min(0.05, (now - last) / 1000); last = now; step(dt, false); raf = requestAnimationFrame(loop); }
  function play() {
    cancelAnimationFrame(raf);
    if (document.hidden) return; // aba em segundo plano: não gasta bateria
    if (reduceMotion.matches) return step(0, true); // um quadro parado
    last = performance.now(); raf = requestAnimationFrame(loop);
  }
  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    if (reduceMotion.matches) step(0, true);
  }

  // ------------------------------------------------------------ eventos da página
  const on = (name, fn) => document.addEventListener("portaria:" + name, (e) => { fn(e.detail || {}); if (reduceMotion.matches) step(0, true); });
  on("view", (d) => { state.view = d.name; state.homeOpen = false; });
  on("enter", () => { state.homeOpen = true; if (!reduceMotion.matches) burst(); });
  on("scan", (d) => { state.scanning = !!d.active; });
  on("hover", (d) => { state.hover = d.side === "l" ? -1 : d.side === "r" ? 1 : 0; });
  on("door", (d) => { state.doorOpen = !!d.open; if (d.open && DOOR_VIEWS[state.view] && !reduceMotion.matches) burst(); });
  on("flash", (d) => {
    state.flash = d.kind === "bad" ? "bad" : "ok"; state.flashUntil = performance.now() + 1400;
    if (state.flash === "bad" && !reduceMotion.matches) state.shake = 1;
  });
  window.addEventListener("pointermove", (e) => { state.pointer.set((e.clientX / window.innerWidth) * 2 - 1, -((e.clientY / window.innerHeight) * 2 - 1)); }, { passive: true });
  window.addEventListener("resize", resize);
  document.addEventListener("visibilitychange", play);
  reduceMotion.addEventListener("change", play);
  canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); cancelAnimationFrame(raf); });
  canvas.addEventListener("webglcontextrestored", play);

  document.body.classList.add("has3d"); // a página troca as linhas de CSS pela porta 3D
  resize();
  if (window.__portariaView) state.view = window.__portariaView; // a página pode ter escolhido a tela antes
  const first = doorTarget();
  cur.pos.copy(target.pos); cur.s = first.s; cur.open = first.open;
  // entrada: a porta aparece já se fechando (ou parada, para quem pede menos movimento)
  if (!reduceMotion.matches && state.view === "home") cur.open = Math.max(first.open, 0.55);
  canvas.classList.add("ready");
  play();
  // só para depuração/testes manuais; snap() desenha o estado final na hora (útil com a aba oculta)
  window.__portariaScene = { state, cur, snap: () => step(0, true) };
}
