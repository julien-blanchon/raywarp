// raywarp viewer page: no build step, no dependency. Frames stream over one WebSocket
// (/api/stream): the page sends its camera when it changes and acknowledges each frame it draws;
// the server pushes frames of the newest camera (a frame: 4-byte length, info JSON, JPEG) and,
// while it is busy, status messages (a scene load, a kernel compile, an OptiX module creation, the
// background warm). Commands go to POST /api/command.

const token = new URLSearchParams(location.search).get('token') || '';
const $ = (id) => document.getElementById(id);
const SIZES = [
  [640, 360],
  [1024, 576],
  [1280, 720],
  [1920, 1080],
  [512, 512],
  [1024, 1024],
];
const BADGES = { runtime: 'R', rebake: 'B', recompile: 'K' };
const KIND_TEXT = {
  kernel: 'kernel compile',
  ptx: 'OptiX PTX compile',
  optix_module: 'OptiX module created',
  other: 'non-renderer module compiled',
};

let state = null; // the server's last state
let camera = null; // {position, yaw, pitch, fov}
let stream = null; // the WebSocket
let dirty = true; // the camera changed: send it
let seq = 0; // the camera messages sent
const sentAt = new Map(); // camera seq -> when it was sent (the camera-to-frame latency)
let latency = 0; // ms from a camera sent to its frame drawn (smoothed)
let pending = null; // the newest frame received and not yet drawn
let drawing = false;
let lastEvent = 0; // the last compile event shown
let speedScale = 1;
let lastFrameAt = 0;
let fps = 0;
const keys = new Set();

async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Viewer-Token': token },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = (await response.json()).error || message;
    } catch {}
    throw new Error(message);
  }
  return response;
}

function banner(text, kind) {
  const el = $('banner');
  el.hidden = !text;
  el.textContent = text || '';
  el.className = kind || '';
}

async function command(body, label) {
  $('loading').hidden = false;
  $('loading').textContent = label || 'applying…';
  try {
    const next = await (await api('/api/command', body)).json();
    applyState(next);
    if (next.message) banner(next.message, 'info');
    else if (!compileWarning()) banner('');
  } catch (e) {
    banner(e.message, 'error');
  } finally {
    $('loading').hidden = true;
    send({ refresh: true });
  }
}

function option(select, value, text, selected) {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = text;
  o.selected = selected;
  select.appendChild(o);
  return o;
}

function fillScenes() {
  const select = $('scene');
  select.replaceChildren();
  const groups = new Map();
  for (const s of state.scenes) {
    if (!groups.has(s.group)) {
      const g = document.createElement('optgroup');
      g.label = s.group;
      groups.set(s.group, g);
      select.appendChild(g);
    }
    option(groups.get(s.group), s.id, s.name, s.id === state.scene);
  }
}

function fillSimple() {
  $('base').replaceChildren();
  for (const b of state.bases) option($('base'), b, b, b === state.base);
  $('view').replaceChildren();
  for (const v of state.views) option($('view'), v, v, v === state.view);
  const sizes = SIZES.some(([w, h]) => w === state.size[0] && h === state.size[1]) ? SIZES : [state.size, ...SIZES];
  $('size').replaceChildren();
  for (const [w, h] of sizes) option($('size'), `${w}x${h}`, `${w} × ${h}`, w === state.size[0] && h === state.size[1]);
  const b = state.bundle;
  const kernels = b.kernels ? `prebuilt kernels ${b.kernels}` : b.arch ? 'NO prebuilt kernels bundle: kernels compile' : 'CPU: kernels compile on first use';
  $('device').textContent = `${state.device} · ${kernels}` + (b.arch ? ` · OptiX module pack: ${b.optix_modules ? 'yes' : 'no'}` : '');
}

function control(row) {
  const send = (value) => command({ type: 'settings', patch: { [row.path]: value } }, `${row.path}…`);
  if (row.type === 'boolean') {
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = row.value;
    input.onchange = () => send(input.checked);
    return input;
  }
  if (row.type === 'enum') {
    const select = document.createElement('select');
    for (const o of row.options) option(select, JSON.stringify(o), String(o), o === row.value);
    select.onchange = () => send(JSON.parse(select.value));
    return select;
  }
  const input = document.createElement('input');
  input.spellcheck = false;
  if (row.type === 'number') {
    input.type = 'number';
    input.step = 'any';
    input.value = row.value;
    input.onchange = () => (input.value.trim() === '' ? banner(`${row.path}: a number is required`, 'error') : send(Number(input.value)));
  } else {
    input.value = row.value ?? '';
    input.onchange = () => send(input.value);
  }
  return input;
}

function fillSettings() {
  const box = $('settings');
  const open = new Set([...box.querySelectorAll('details[open]')].map((d) => d.dataset.group));
  const filter = $('filter').value.trim().toLowerCase();
  const onlyChanged = $('only-changed').checked;
  const changed = new Set(state.changed);
  box.replaceChildren();
  const groups = new Map();
  for (const row of state.settings) {
    if (filter && !row.path.toLowerCase().includes(filter) && !row.description.toLowerCase().includes(filter)) continue;
    if (onlyChanged && !changed.has(row.path)) continue;
    if (!groups.has(row.group)) {
      const d = document.createElement('details');
      d.dataset.group = row.group;
      d.open = open.has(row.group) || Boolean(filter) || onlyChanged;
      const s = document.createElement('summary');
      s.textContent = row.group;
      d.appendChild(s);
      groups.set(row.group, d);
      box.appendChild(d);
    }
    const line = document.createElement('div');
    line.className = 'setting' + (changed.has(row.path) ? ' changed' : '');
    line.title = `${row.path}: ${row.description}`;
    const badge = document.createElement('span');
    badge.className = `badge ${row.change}`;
    badge.textContent = BADGES[row.change];
    const label = document.createElement('label');
    label.textContent = row.path.split('.').slice(1).join('.');
    line.append(badge, label, control(row));
    groups.get(row.group).appendChild(line);
  }
  for (const d of groups.values()) {
    const n = d.querySelectorAll('.changed').length;
    if (n) d.querySelector('summary').textContent = `${d.dataset.group} (${n} changed)`;
  }
}

function applyState(next) {
  const sceneChanged = !state || state.scene !== next.scene || !camera;
  state = next;
  if (sceneChanged) camera = structuredClone(next.camera);
  fillScenes();
  fillSimple();
  fillSettings();
  $('fov').value = camera.fov;
  $('fov-value').textContent = `${Math.round(camera.fov)}°`;
  showCompiles(next.compiles, next.kernels);
}

function compileWarning() {
  return state && state.compiles && state.compiles.kernel + state.compiles.ptx > 0;
}

function showCompiles(counts, kernels) {
  if (!counts) return;
  const el = $('compiles');
  const bad = counts.kernel + counts.ptx;
  el.className = bad ? 'bad' : counts.optix_module ? 'warn' : 'good';
  el.textContent =
    `compiles: ${counts.kernel} kernel · ${counts.ptx} PTX · ${counts.optix_module} OptiX module creations` +
    (counts.other ? ` · ${counts.other} other modules (simulator)` : '');
  if (kernels !== undefined) {
    $('kernels').textContent = kernels ? `kernels: ${kernels}` : '';
    $('kernels').className = kernels.startsWith('own') ? 'bad' : 'muted';
  }
  if (state) state.compiles = counts;
}

let polling = false; // one events request at a time (frames and statuses both ask)
async function pollEvents() {
  if (polling) return;
  polling = true;
  try {
    await fetchEvents();
  } catch (e) {
    banner(`events: ${e.message}`, 'error');
  } finally {
    polling = false;
  }
}

async function fetchEvents() {
  const data = await (await api(`/api/events?since=${lastEvent}`)).json();
  const list = $('events');
  for (const e of data.events) {
    lastEvent = Math.max(lastEvent, e.seq);
    const li = document.createElement('li');
    li.className = e.kind === 'kernel' || e.kind === 'ptx' ? 'bad' : 'warn';
    const when = new Date(e.time * 1000).toLocaleTimeString();
    li.textContent =
      `${when} ${e.background ? 'background warm: ' : ''}${KIND_TEXT[e.kind]}: ${e.name}` + (e.seconds ? ` (${e.seconds.toFixed(2)} s)` : '');
    list.prepend(li);
    if (e.kind === 'kernel' || e.kind === 'ptx') {
      banner(`${KIND_TEXT[e.kind]}: ${e.name} (${e.seconds.toFixed(1)} s): this configuration is outside the prebuilt kernels`, 'error');
      $('events-box').open = true;
    }
  }
  $('event-count').textContent = String(lastEvent);
  showCompiles(data.compiles);
}

function moveCamera(dt) {
  const fast = keys.has('shift') ? 4 : 1;
  const speed = (state?.loaded?.extent ?? 2) * 0.6 * speedScale * fast * dt;
  const { yaw, pitch } = camera;
  const forward = [Math.sin(yaw) * Math.cos(pitch), Math.cos(yaw) * Math.cos(pitch), Math.sin(pitch)];
  const right = [Math.cos(yaw), -Math.sin(yaw), 0];
  const step = [0, 0, 0];
  const add = (v, s) => v.forEach((c, i) => (step[i] += c * s));
  if (keys.has('w')) add(forward, 1);
  if (keys.has('s')) add(forward, -1);
  if (keys.has('d')) add(right, 1);
  if (keys.has('a')) add(right, -1);
  if (keys.has('e')) add([0, 0, 1], 1);
  if (keys.has('q')) add([0, 0, 1], -1);
  if (step.every((c) => c === 0)) return false;
  camera.position = camera.position.map((c, i) => c + step[i] * speed);
  return true;
}

function send(message) {
  if (stream && stream.readyState === WebSocket.OPEN) stream.send(JSON.stringify(message));
}

function openStream() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  stream = new WebSocket(`${scheme}://${location.host}/api/stream?token=${encodeURIComponent(token)}`);
  stream.binaryType = 'arraybuffer';
  stream.onopen = () => {
    dirty = true;
    send({ live: $('live').checked });
  };
  stream.onmessage = (e) => (typeof e.data === 'string' ? onText(JSON.parse(e.data)) : onFrame(e.data));
  stream.onclose = () => {
    banner('frame stream closed: reconnecting…', 'error');
    setTimeout(openStream, 1000);
  };
}

function onText(message) {
  if (message.type === 'status') showStatus(message);
  else if (message.type === 'error') banner(message.error, 'error');
}

function onFrame(buffer) {
  const n = new DataView(buffer).getUint32(0);
  const info = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
  pending = { info, jpeg: new Blob([new Uint8Array(buffer, 4 + n)], { type: 'image/jpeg' }) };
  if (!drawing) draw();
}

// Draws the newest frame received (older ones are skipped), then acknowledges it (and every
// frame before it).
async function draw() {
  drawing = true;
  while (pending) {
    const { info, jpeg } = pending;
    pending = null;
    try {
      const bitmap = await createImageBitmap(jpeg);
      const canvas = $('canvas');
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      bitmap.close();
    } catch (e) {
      banner(`frame not shown: ${e.message}`, 'error');
    }
    send({ ack: info.id });
    shown(info);
  }
  drawing = false;
}

function shown(info) {
  const now = performance.now();
  if (lastFrameAt) fps = fps ? fps * 0.9 + 100 / (now - lastFrameAt) : 1000 / (now - lastFrameAt);
  lastFrameAt = now;
  if (sentAt.has(info.seq)) {
    const ms = now - sentAt.get(info.seq);
    latency = latency ? latency * 0.8 + ms * 0.2 : ms;
    for (const k of sentAt.keys()) if (k <= info.seq) sentAt.delete(k);
  }
  $('hud').textContent =
    `${fps.toFixed(1)} fps · render ${info.render_ms.toFixed(1)} ms (avg ${info.render_ms_avg.toFixed(1)})` +
    ` · read ${info.read_ms.toFixed(1)} · jpeg ${info.encode_ms.toFixed(1)} ms · ${(info.bytes / 1024).toFixed(0)} KiB` +
    ` · ${info.size[0]}×${info.size[1]} · camera→frame ${latency.toFixed(0)} ms` +
    ` · window ${info.inflight}/${info.window} rtt ${info.rtt_ms.toFixed(0)} ms` +
    (info.pending ? ` · ${info.pending} modules loading` : '');
  showCompiles(info.compiles, info.kernels);
  if (Object.values(info.compiles).reduce((a, b) => a + b, 0) > lastEvent) pollEvents();
  window.viewerFrames = (window.viewerFrames || 0) + 1; // read by the screenshot script
}

const ACTIVITY = {
  kernel: 'compiling kernel module',
  ptx: 'compiling OptiX programs (PTX)',
  optix_module: 'creating OptiX module',
  other: 'compiling non-renderer module',
};

// What the server is busy with, live: the render thread's work, each compile or module creation
// running (with its estimate), the background warm.
function showStatus(status) {
  const lines = [];
  if (status.busy) lines.push(`${status.busy.what === 'frame' ? 'rendering' : status.busy.what}… ${status.busy.seconds.toFixed(1)} s`);
  for (const r of status.running) {
    const est = r.estimate ? ` (est. ${r.estimate.toFixed(0)} s)` : '';
    lines.push(`${r.background ? 'warm: ' : ''}${ACTIVITY[r.kind]} ${r.name}: ${r.seconds.toFixed(1)} s${est}`);
  }
  const el = $('activity');
  el.hidden = lines.length === 0;
  el.textContent = lines.join('\n');
  showWarm(status.warm);
  showCompiles(status.compiles);
  if (status.events > lastEvent) pollEvents();
}

function showWarm(warm) {
  const el = $('warm');
  if (!warm || warm.scene === undefined) {
    el.textContent = '';
    return;
  }
  if (warm.running) {
    el.className = 'warn';
    el.textContent = `background warm (views, settings one change away): ${warm.seconds.toFixed(0)} s`;
  } else if (warm.error) {
    el.className = 'bad';
    el.textContent = `background warm failed: ${warm.error}`;
  } else {
    el.className = 'good';
    const modules = warm.modules === undefined ? '' : `${warm.modules} OptiX modules, `;
    el.textContent = `warm: ${modules}views and one-change settings ready (${warm.seconds.toFixed(0)} s)`;
  }
}

let previous = performance.now();
function tick(now) {
  const dt = Math.min((now - previous) / 1000, 0.1);
  previous = now;
  if (camera && moveCamera(dt)) dirty = true;
  if (camera && dirty && stream && stream.readyState === WebSocket.OPEN) {
    dirty = false;
    seq += 1;
    sentAt.set(seq, performance.now());
    send({ camera, seq, live: $('live').checked });
  }
  requestAnimationFrame(tick);
}

function bindInputs() {
  $('scene').onchange = () => command({ type: 'scene', id: $('scene').value }, 'loading scene…');
  $('path-form').onsubmit = (e) => {
    e.preventDefault();
    command({ type: 'path', path: $('path').value.trim() }, 'loading scene…');
  };
  $('base').onchange = () => command({ type: 'base', name: $('base').value }, 'switching settings…');
  $('reset').onclick = () => command({ type: 'base', name: state.base }, 'resetting…');
  $('view').onchange = () => command({ type: 'view', name: $('view').value });
  $('size').onchange = () => {
    const [width, height] = $('size').value.split('x').map(Number);
    command({ type: 'size', width, height }, 'resizing…');
  };
  // A new field of view is a new renderer: applied on release, not on every slider step.
  $('fov').oninput = () => ($('fov-value').textContent = `${$('fov').value}°`);
  $('fov').onchange = () => {
    camera.fov = Number($('fov').value);
    dirty = true;
  };
  $('home').onclick = home;
  $('live').onchange = () => send({ live: $('live').checked });
  $('filter').oninput = fillSettings;
  $('only-changed').onchange = fillSettings;
  const stage = $('stage');
  let drag = null;
  stage.onpointerdown = (e) => {
    stage.focus();
    drag = { x: e.clientX, y: e.clientY };
    stage.setPointerCapture(e.pointerId);
  };
  stage.onpointermove = (e) => {
    if (!drag || !camera) return;
    camera.yaw += (e.clientX - drag.x) * 0.004;
    camera.pitch = Math.max(-1.56, Math.min(1.56, camera.pitch - (e.clientY - drag.y) * 0.004));
    drag = { x: e.clientX, y: e.clientY };
    dirty = true;
  };
  stage.onpointerup = () => (drag = null);
  stage.onwheel = (e) => {
    e.preventDefault();
    speedScale = Math.max(0.02, Math.min(50, speedScale * (e.deltaY < 0 ? 1.25 : 0.8)));
    banner(`fly speed × ${speedScale.toFixed(2)}`, 'info');
  };
  stage.onkeydown = (e) => {
    const k = e.key.toLowerCase();
    if (k === 'h') return home();
    if ('wasdqe'.includes(k) || k === 'shift') {
      keys.add(k);
      e.preventDefault();
    }
  };
  stage.onkeyup = (e) => keys.delete(e.key.toLowerCase());
  stage.onblur = () => keys.clear();
}

async function home() {
  await command({ type: 'home' });
  camera = structuredClone(state.camera);
  dirty = true;
}

async function start() {
  bindInputs();
  try {
    applyState(await (await api('/api/state')).json());
  } catch (e) {
    banner(e.message, 'error');
    return;
  }
  openStream(); // first: its statuses show the scene load and its compiles
  camera = null; // the scene's own view once it is loaded
  await command({ type: 'scene', id: state.scene }, 'loading scene…');
  camera = camera || structuredClone(state.camera);
  dirty = true;
  await pollEvents();
  requestAnimationFrame(tick);
}

start();
