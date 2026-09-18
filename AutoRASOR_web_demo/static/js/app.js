// AutoRASOR demo — wiring: store, API, controls, timeline.

import { LAYERS, LAYER_BY_ID } from './layers.js';
import { Viewport } from './viewport.js';

const $ = (sel) => document.querySelector(sel);

// ── store ────────────────────────────────────────────────────

const store = {
  serial: 0,
  micrograph: { ready: false },
  state: null,
  images: {},
  tiles: new Map(),
  focusIndex: null,
  hoverIndex: null,
  busy: false,
  opts: {
    cmap: 'viridis',
    nearest: true,
    grid: true,
    markers: true,
    stepNumbers: true,
    sharedScale: true,
    evrWeighted: true,
    sync: true,
    follow: true,
    latentColor: 'gt',
  },
  tileImage(idx) {
    if (!this.tiles.has(idx)) {
      const img = new Image();
      img.src = `/img/tile/${idx}.png?v=${this.serial}`;
      img.onload = () => render();
      this.tiles.set(idx, img);
    }
    return this.tiles.get(idx);
  },
};

// ── api ──────────────────────────────────────────────────────

async function getJSON(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 6000);
}

function setStatus(text, cls = '') {
  const el = $('#status-text');
  el.textContent = text;
  el.className = 'status ' + cls;
}

// ── viewports ────────────────────────────────────────────────

// Declared before construction: a Viewport fits itself on creation, which calls
// back into these hooks while `left`/`right` are still being assigned.
let left = null;
let right = null;

const hooks = {
  onViewChange(vp) {
    if (!left || !right || !store.opts.sync) return;
    const other = vp === left ? right : left;
    if (other.space === vp.space) other.setView(vp.view);
    updateSyncBadges();
  },
  onHover(idx, vp) {
    if (store.hoverIndex === idx) return;
    store.hoverIndex = idx;
    updateHoverLine();
    render();
  },
  onPick(idx) {
    if (idx == null) return;
    store.focusIndex = idx;
    $('#opt-follow').checked = false;
    store.opts.follow = false;
    render();
  },
  onLayerChange() {
    updateSyncBadges();
  },
};

left = new Viewport(document.querySelector('.viewport[data-side=left]'), 'left', store, hooks);
right = new Viewport(document.querySelector('.viewport[data-side=right]'), 'right', store, hooks);

function render() {
  if (left) left.invalidate();
  if (right) right.invalidate();
}

function updateSyncBadges() {
  if (!left || !right) return;
  const linked = store.opts.sync && left.space === right.space;
  for (const vp of [left, right]) {
    vp.els.space.textContent = linked ? `${vp.space} · synced` : vp.space;
    vp.els.space.classList.toggle('synced', linked);
  }
}

// ── layer selects ────────────────────────────────────────────

function fillLayerSelect(sel, initial) {
  const groups = {};
  for (const layer of LAYERS) (groups[layer.group] ||= []).push(layer);
  for (const [name, items] of Object.entries(groups)) {
    const og = document.createElement('optgroup');
    og.label = name;
    for (const layer of items) {
      const opt = document.createElement('option');
      opt.value = layer.id;
      opt.textContent = layer.label;
      og.append(opt);
    }
    sel.append(og);
  }
  sel.value = initial;
}

fillLayerSelect($('#layer-left'), 'lowmag');
fillLayerSelect($('#layer-right'), 'highmag');
$('#layer-left').addEventListener('change', (e) => { left.setLayer(e.target.value); updateSyncBadges(); });
$('#layer-right').addEventListener('change', (e) => { right.setLayer(e.target.value); updateSyncBadges(); });

// ── display options ──────────────────────────────────────────

const OPT_BINDINGS = [
  ['#opt-sync', 'sync'], ['#opt-nn', 'nearest'], ['#opt-grid', 'grid'],
  ['#opt-marks', 'markers'], ['#opt-steps', 'stepNumbers'],
  ['#opt-shared', 'sharedScale'], ['#opt-evr', 'evrWeighted'], ['#opt-follow', 'follow'],
];
for (const [sel, key] of OPT_BINDINGS) {
  $(sel).addEventListener('change', (e) => {
    store.opts[key] = e.target.checked;
    if (key === 'sync' && e.target.checked && left.space === right.space) right.setView(left.view);
    if (key === 'follow' && e.target.checked) syncFocusToSelection();
    updateSyncBadges();
    render();
  });
}
$('#cmap').addEventListener('change', (e) => { store.opts.cmap = e.target.value; render(); });
$('#latent-color').addEventListener('change', (e) => { store.opts.latentColor = e.target.value; render(); });

// ── micrograph controls ──────────────────────────────────────

const sliderPairs = [
  ['#mic-blur', '#mic-blur-out'], ['#mic-noise', '#mic-noise-out'],
  ['#mic-lblur', '#mic-lblur-out'], ['#budget', '#budget-out'],
];
for (const [inp, out] of sliderPairs) {
  const i = $(inp), o = $(out);
  const sync = () => { o.value = i.value; };
  i.addEventListener('input', sync);
  sync();
}

function micrographBody(seed) {
  return {
    seed,
    keep_gpr: true,
    global_blur_sigma: Number($('#mic-blur').value),
    gaussian_noise_std: Number($('#mic-noise').value),
    local_blur_max_sigma: Number($('#mic-lblur').value),
  };
}

$('#btn-new-mic').addEventListener('click', () => {
  const seed = Number($('#mic-seed').value) + 1;
  $('#mic-seed').value = seed;
  buildMicrograph(seed);
});
$('#btn-rebuild').addEventListener('click', () => buildMicrograph(Number($('#mic-seed').value)));

async function buildMicrograph(seed) {
  if (store.busy) return;
  setBusy(true, 'building micrograph…');
  $('#build-progress').hidden = false;
  try {
    const res = await postJSON('/api/micrograph', micrographBody(seed));
    if (!res.started) { toast(res.reason || 'build refused'); setBusy(false); return; }
    await pollProgress();
    await loadMicrograph();
  } catch (err) {
    toast(String(err));
    setStatus('build failed', 'err');
  } finally {
    $('#build-progress').hidden = true;
    setBusy(false);
  }
}

async function pollProgress() {
  for (;;) {
    const p = await getJSON('/api/progress');
    $('#build-bar').style.width = `${Math.round(p.frac * 100)}%`;
    $('#build-msg').textContent = p.message || '';
    setStatus(p.message || 'working…', 'busy');
    if (p.error) throw new Error(p.error.split('\n').slice(-1)[0]);
    if (p.status !== 'building') return;
    await new Promise((r) => setTimeout(r, 350));
  }
}

async function loadMicrograph() {
  const m = await getJSON('/api/micrograph');
  if (!m.ready) return;
  store.micrograph = m;
  store.serial = m.serial;
  store.tiles.clear();
  store.images = {};
  for (const key of ['lowmag', 'lowmag_clean', 'highmag']) {
    const img = new Image();
    img.src = `/img/${key}.png?v=${m.serial}`;
    img.onload = () => render();
    store.images[key] = img;
  }
  $('#budget').max = String(m.n_points);
  $('#mic-seed').value = m.seed;
  await refreshState();
  setStatus(`micrograph #${m.serial} · seed ${m.seed} · built in ${m.build_seconds}s`);
}

// ── policy / run controls ────────────────────────────────────

const POLICY_HINTS = {
  ambiguity: 'Fits a Matérn-5/2 ARD GP to the ambiguity of what has been captured, then '
    + 'maximizes the acquisition function over the unsampled patches. Needs ≥5 captures before '
    + 'the GP is used; below that the engine falls back to random picks.',
  lfps: 'Greedy farthest-point walk in the low-mag DINOv3 latent space. Needs no captures and '
    + 'no surrogate — the whole ordering is known up front. The GP is still fitted so you can '
    + 'watch what LFPS coverage teaches it.',
  random: 'Uniform random picks from the unsampled patches — the control condition.',
};

function policyUI() {
  const p = $('#policy').value;
  $('#row-acq').style.display = p === 'ambiguity' ? '' : 'none';
  $('#row-warmup').style.display = p === 'ambiguity' ? '' : 'none';
  $('#policy-hint').textContent = POLICY_HINTS[p];
}
$('#policy').addEventListener('change', policyUI);
policyUI();

function runBody() {
  return {
    policy: $('#policy').value,
    acq: $('#acq').value,
    budget: Number($('#budget').value),
    batch_size: Number($('#batch').value),
    warmup_seeds: Number($('#warmup').value),
    seed: Number($('#run-seed').value),
    beta: 1.0,
  };
}

$('#btn-start').addEventListener('click', async () => {
  await guard('starting run…', async () => {
    applyState(await postJSON('/api/run', runBody()));
  });
});

$('#btn-reset-gpr').addEventListener('click', async () => {
  await guard('resetting surrogate…', async () => {
    applyState(await postJSON('/api/reset_gpr', {}));
    toast('GPR reset — all carried observations discarded');
  });
});

$('#btn-next').addEventListener('click', () => step('next'));
$('#btn-prev').addEventListener('click', () => step('prev'));
$('#btn-first').addEventListener('click', () => step('start'));
$('#btn-end').addEventListener('click', () => step('end'));

async function step(dir) {
  const label = dir === 'end' ? 'running to budget…' : 'stepping…';
  await guard(label, async () => {
    applyState(await postJSON('/api/step', { dir, max_steps: 400 }));
  });
}

async function guard(label, fn) {
  if (store.busy || !store.micrograph.ready) return;
  setBusy(true, label);
  try {
    await fn();
    setStatus(idleStatus());
  } catch (err) {
    toast(String(err));
    setStatus('request failed', 'err');
  } finally {
    setBusy(false);
  }
}

function idleStatus() {
  const s = store.state;
  if (!s) return 'ready';
  return `step ${s.step} · ${s.captures.length}/${s.run.budget} captures · GP fit ${s.fit_seconds}s`;
}

function setBusy(on, label) {
  store.busy = on;
  if (on && label) setStatus(label, 'busy');
  for (const id of ['#btn-next', '#btn-prev', '#btn-first', '#btn-end', '#btn-start',
                    '#btn-reset-gpr', '#btn-new-mic', '#btn-rebuild']) {
    $(id).disabled = on;
  }
  if (!on) updateStepButtons();
}

async function refreshState() {
  for (let attempt = 0; attempt < 6; attempt++) {
    const s = await getJSON('/api/state');
    if (s.ready) { applyState(s); return; }
    await new Promise((r) => setTimeout(r, 300));
  }
}

// ── state application ────────────────────────────────────────

function applyState(s) {
  if (!s || !s.ready) return;
  store.state = s;
  $('#policy').value = s.run.policy;
  $('#acq').value = s.run.acq;
  $('#budget').value = s.run.budget;
  $('#budget-out').value = s.run.budget;
  $('#batch').value = s.run.batch_size;
  $('#warmup').value = s.run.warmup_seeds;
  $('#run-seed').value = s.run.seed;
  policyUI();

  $('#chip-engine').textContent =
    `${s.engine.kernel} · ${s.engine.fit_mode} · ${s.engine.noise_mode} · k=${s.engine.k_neighbors}`;

  if (store.opts.follow) syncFocusToSelection();
  updateStepButtons();
  updateTimeline();
  updateMetrics();
  updatePickBox();
  updateHoverLine();
  render();
}

function syncFocusToSelection() {
  const s = store.state;
  if (!s) return;
  if (s.pending.length) store.focusIndex = s.pending[0];
  else if (s.captures.length) store.focusIndex = s.captures[s.captures.length - 1].idx;
}

function updateStepButtons() {
  const s = store.state;
  if (!s) return;
  $('#btn-prev').disabled = store.busy || s.cursor === 0;
  $('#btn-first').disabled = store.busy || s.cursor === 0;
  $('#btn-next').disabled = store.busy || !s.can_advance;
  $('#btn-end').disabled = store.busy || !s.can_advance;

  $('#step-now').textContent = `step ${s.step}`;
  $('#step-frontier').textContent =
    `frontier ${s.frontier_step}${s.n_carried ? ` · ${s.n_carried} carried obs` : ''}`;

  const n = s.captures.length;
  $('#budget-bar').style.width = `${Math.round(100 * n / Math.max(s.run.budget, 1))}%`;
  $('#budget-msg').textContent = `${n} / ${s.run.budget} captures`;

  const gp = s.gp;
  $('#gp-hint').textContent = !gp.active
    ? `Surrogate idle: ${s.engine.n_observed} observation(s), needs 5.`
    : gp.flat
      ? 'GP fitted but its posterior is still essentially flat — the ARD lengthscales have not localized yet.'
      : `GP fitted on ${s.engine.n_observed} observations.`;
}

function fmt(v, digits = 3) {
  if (v == null || !Number.isFinite(v)) return '—';
  return Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(digits);
}

function updateMetrics() {
  const s = store.state;
  const m = (s && s.metrics) || {};
  const rows = [
    ['captured', `${Math.round(m.n_captured ?? 0)} (${fmt(m.pct_captured, 1)}%)`],
    ['Spearman ρ', fmt(m.spearman)],
    ['top-20 recall', m.recall == null ? '—' : `${fmt(m.recall, 0)}%`],
    ['top-20 found', m.discovery == null ? '—' : `${fmt(m.discovery, 0)}%`],
    ['best GT found', `${fmt(m.best_gt_found)} / ${fmt(m.gt_max)}`],
    ['covering radius', fmt(m.covering_radius)],
    ['mean min dist', fmt(m.mean_min_dist)],
    ['GP fit time', s ? `${fmt(s.fit_seconds, 2)}s` : '—'],
  ];
  const host = $('#metrics');
  host.replaceChildren();
  for (const [label, value] of rows) {
    const div = document.createElement('div');
    div.className = 'metric';
    const sp = document.createElement('span');
    sp.textContent = label;
    const b = document.createElement('b');
    b.textContent = value;
    div.append(sp, b);
    host.append(div);
  }
}

function rc(idx) {
  const g = store.micrograph.grid;
  return [Math.floor(idx / g.cols), idx % g.cols];
}

function updatePickBox() {
  const s = store.state;
  const box = $('#pick-box');
  if (!s) { box.textContent = 'no run'; return; }
  const parts = [];
  const thisStep = s.captures.filter((c) => c.step === s.step && c.kind === 'policy');
  if (thisStep.length) {
    parts.push(`captured @${s.step}: ` + thisStep.map((c) => {
      const [r, c2] = rc(c.idx);
      return `#${c.idx} (r${r} c${c2})`;
    }).join(', '));
  } else if (s.step === 0) {
    parts.push(s.captures.length ? `warmup: ${s.captures.length} LFPS seeds` : 'no captures yet');
  }
  if (s.pending.length) {
    parts.push('next → ' + s.pending.map((idx) => {
      const [r, c] = rc(idx);
      const gt = store.micrograph.gt_ambiguity[idx];
      const mean = s.gp.mean ? s.gp.mean[idx] : null;
      const sd = s.gp.std ? s.gp.std[idx] : null;
      const pred = mean == null ? '' : `, GP ${fmt(mean)}±${fmt(sd)}`;
      return `#${idx} (r${r} c${c}) GT ${fmt(gt)}${pred}`;
    }).join(' | '));
  } else {
    parts.push('budget reached');
  }
  box.textContent = parts.join('\n');
}

function updateHoverLine() {
  const idx = store.hoverIndex;
  const line = $('#hoverline');
  if (idx == null || !store.micrograph.ready) {
    line.textContent = 'hover a patch for its numbers · click to pin it as the selected tile';
    return;
  }
  const s = store.state;
  const [r, c] = rc(idx);
  const bits = [`patch #${idx} (r${r} c${c})`];
  bits.push(`GT amb ${fmt(store.micrograph.gt_ambiguity[idx])}`);
  if (s && s.gp.mean) bits.push(`GPR ${fmt(s.gp.mean[idx])} ± ${fmt(s.gp.std[idx])}`);
  if (s && s.gp.acq) bits.push(`acq ${fmt(s.gp.acq[idx])}`);
  bits.push(`AF ${fmt(store.micrograph.af_gt[idx], 3)}`);
  bits.push(`mob ${fmt(store.micrograph.mob_gt[idx], 3)}`);
  if (s) {
    const cap = s.captures.find((x) => x.idx === idx);
    if (cap) bits.push(cap.kind === 'warmup' ? 'captured in warmup' : `captured at step ${cap.step}`);
    if (s.obs_ambiguity && s.obs_ambiguity[idx] != null) {
      bits.push(`observed amb ${fmt(s.obs_ambiguity[idx])}`);
    }
    if (s.pending.includes(idx)) bits.push('← next pick');
  }
  line.textContent = bits.join(' · ');
}

// ── timeline ─────────────────────────────────────────────────

function updateTimeline() {
  const s = store.state;
  const host = $('#timeline');
  host.replaceChildren();
  if (!s) return;

  const byStep = new Map();
  for (const cap of s.captures) {
    if (!byStep.has(cap.step)) byStep.set(cap.step, []);
    byStep.get(cap.step).push(cap);
  }
  if (!byStep.has(0)) byStep.set(0, []);

  const steps = [...byStep.keys()].sort((a, b) => a - b);
  for (const step of steps) {
    const caps = byStep.get(step);
    const chip = document.createElement('button');
    const kind = step === 0 ? 'warmup' : 'policy';
    chip.className = `tl-chip ${kind}${step === s.step ? ' current' : ''}`;
    const label = step === 0 ? (caps.length ? `W×${caps.length}` : 'start') : String(step);
    const sub = document.createElement('span');
    sub.className = 'n';
    sub.textContent = caps.length ? caps.map((c) => `#${c.idx}`).join(' ') : '—';
    chip.append(label, sub);
    chip.title = caps.map((c) => {
      const [r, cc] = rc(c.idx);
      return `patch #${c.idx} (r${r} c${cc}) · GT ${fmt(store.micrograph.gt_ambiguity[c.idx])}`;
    }).join('\n') || 'run start';
    chip.addEventListener('click', () => gotoStep(step));
    chip.addEventListener('mouseenter', () => {
      if (caps.length) { store.hoverIndex = caps[0].idx; updateHoverLine(); render(); }
    });
    chip.addEventListener('mouseleave', () => { store.hoverIndex = null; updateHoverLine(); render(); });
    host.append(chip);
  }

  if (s.pending.length) {
    const chip = document.createElement('button');
    chip.className = 'tl-chip pending';
    const sub = document.createElement('span');
    sub.className = 'n';
    sub.textContent = s.pending.map((i) => `#${i}`).join(' ');
    chip.append('next', sub);
    chip.title = 'the pick this posterior recommends — press ▶ to capture it';
    chip.addEventListener('click', () => step('next'));
    chip.addEventListener('mouseenter', () => {
      store.hoverIndex = s.pending[0]; updateHoverLine(); render();
    });
    chip.addEventListener('mouseleave', () => { store.hoverIndex = null; updateHoverLine(); render(); });
    host.append(chip);
  }

  const current = host.querySelector('.tl-chip.current');
  if (current) current.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

async function gotoStep(step) {
  await guard('seeking…', async () => {
    applyState(await postJSON('/api/step', { dir: 'goto', step }));
  });
}

// ── panel + keyboard ─────────────────────────────────────────

$('#panel-toggle').addEventListener('click', () => {
  const panel = $('#panel');
  panel.classList.toggle('collapsed');
  $('#panel-toggle').textContent = panel.classList.contains('collapsed') ? '▸ controls' : '▾ controls';
});

document.addEventListener('visibilitychange', () => { if (!document.hidden) render(); });

// Handy from the browser console: __demo.store.state, __demo.left.view, …
window.__demo = { store, get left() { return left; }, get right() { return right; }, render };

window.addEventListener('keydown', (e) => {
  const target = e.target;
  if (target instanceof Element && target.matches('input, select, textarea, button')) return;
  if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); step('next'); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); step('prev'); }
  else if (e.key === 'f') { left.fit(); right.fit(); }
});

// ── boot ─────────────────────────────────────────────────────

(async function boot() {
  updateSyncBadges();
  try {
    const info = await getJSON('/api/bootstrap');
    $('#chip-device').textContent =
      `${info.device} · DINOv3 layer ${info.dino_layer}${info.use_tta ? ' · D4 TTA' : ''}`;
    for (const [key, sel] of [['global_blur_sigma', '#mic-blur'],
                              ['gaussian_noise_std', '#mic-noise'],
                              ['local_blur_max_sigma', '#mic-lblur'],
                              ['seed', '#mic-seed']]) {
      if (info.defaults[key] != null) $(sel).value = info.defaults[key];
    }
    for (const [inp, out] of sliderPairs) $(out).value = $(inp).value;

    if (info.micrograph && info.micrograph.ready) {
      await loadMicrograph();
    } else if (info.building) {
      setBusy(true, 'building micrograph…');
      $('#build-progress').hidden = false;
      await pollProgress();
      await loadMicrograph();
      $('#build-progress').hidden = true;
      setBusy(false);
    } else {
      await buildMicrograph(Number($('#mic-seed').value));
    }
  } catch (err) {
    toast(String(err));
    setStatus('startup failed', 'err');
  }
})();
