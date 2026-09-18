// A pan/zoom canvas viewport.
//
// Everything is drawn in a unit square: the 14x14 field, a single tile, and the
// latent scatter all map onto [0,1]^2, so one transform ({scale, cx, cy}) works
// for every layer and two viewports in the same space can share it verbatim.
// Scaling is nearest-neighbour by default — no invented pixels between patches.

import { buildSource, sourceKey, LAYER_BY_ID } from './layers.js';
import { colorbarCanvas, table as cmapTable } from './colormaps.js';

const PAD = 0.94;             // fit leaves a small margin
const MIN_SCALE = 0.05;
const MAX_SCALE = 6000;

const COLOR = {
  warmup: '#35d6c8',
  policy: '#ffb44d',
  pending: '#ffffff',
  hover: '#ffffff',
  focus: '#4da3ff',
  grid: 'rgba(255,255,255,0.16)',
};

// `fitted` means "auto-framed": the view re-fits on resize and on layer change
// until the user pans or zooms it themselves.
function newView() {
  return { scale: 1, cx: 0.5, cy: 0.5, fitted: true };
}

// ── mip pyramid ──────────────────────────────────────────────
// Nearest-neighbour is the right call when magnifying (a patch must read as a
// patch, not as an interpolated guess), but it aliases badly when a 3136px
// montage is squeezed into 300px. Pre-filtered halvings fix the shrink without
// touching the magnified case: the last step is still a nearest-neighbour blit.

const MIPS = new WeakMap();

function mipChain(node) {
  let chain = MIPS.get(node);
  if (chain) return chain;
  chain = [node];
  let w = node.naturalWidth || node.width;
  let h = node.naturalHeight || node.height;
  let src = node;
  while (w > 64 && h > 64) {
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const c = cv.getContext('2d');
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = 'high';
    c.drawImage(src, 0, 0, w, h);
    chain.push(cv);
    src = cv;
  }
  MIPS.set(node, chain);
  return chain;
}

/** Smallest level still at least as wide as the on-screen size. */
function pickMip(node, targetW) {
  const native = node.naturalWidth || node.width;
  if (!native || targetW >= native * 0.9) return node;
  let best = node;
  for (const level of mipChain(node)) {
    if ((level.naturalWidth || level.width) >= targetW) best = level;
    else break;
  }
  return best;
}

export class Viewport {
  constructor(root, side, store, hooks = {}) {
    this.root = root;
    this.side = side;
    this.store = store;
    this.hooks = hooks;
    this.canvas = root.querySelector('[data-role=canvas]');
    this.ctx = this.canvas.getContext('2d');
    this.els = {
      label: root.querySelector('[data-role=label]'),
      space: root.querySelector('[data-role=space]'),
      zoom: root.querySelector('[data-role=zoom]'),
      scalebar: root.querySelector('[data-role=scalebar]'),
      empty: root.querySelector('[data-role=empty]'),
      fit: root.querySelector('[data-role=fit]'),
    };

    this.layerId = side === 'left' ? 'lowmag' : 'highmag';
    this.views = { field: newView(), tile: newView(), latent: newView() };
    this.source = null;
    this.sourceKey = null;
    this.latentBounds = null;
    this.dpr = 1;
    this.w = 1;
    this.h = 1;
    this._pending = false;
    this._rafId = 0;
    this._timer = 0;
    this._drag = null;

    this._bindEvents();
    this._observeSize();
  }

  // ── geometry ───────────────────────────────────────────────
  get space() { return (LAYER_BY_ID[this.layerId] || {}).space || 'field'; }
  get view() { return this.views[this.space]; }

  setView(v) {
    const view = this.view;
    view.scale = v.scale; view.cx = v.cx; view.cy = v.cy; view.fitted = v.fitted;
    this.invalidate();
  }

  fit() {
    const view = this.view;
    view.scale = Math.min(this.w, this.h) * PAD;
    view.cx = 0.5; view.cy = 0.5; view.fitted = true;
    this._changed();
  }

  toScreen(u, v) {
    const { scale, cx, cy } = this.view;
    return [(u - cx) * scale + this.w / 2, (v - cy) * scale + this.h / 2];
  }

  toUnit(px, py) {
    const { scale, cx, cy } = this.view;
    return [(px - this.w / 2) / scale + cx, (py - this.h / 2) / scale + cy];
  }

  /** Where the source sits inside the unit square (keeps non-square sources honest). */
  sourceRect() {
    const src = this.source;
    let sw = 1, sh = 1;
    const node = src && (src.image || src.canvas);
    if (node) {
      sw = node.naturalWidth || node.width;
      sh = node.naturalHeight || node.height;
    }
    const a = sw / sh;
    if (a >= 1) return { x: 0, y: (1 - 1 / a) / 2, w: 1, h: 1 / a };
    return { x: (1 - a) / 2, y: 0, w: a, h: 1 };
  }

  // ── lifecycle ──────────────────────────────────────────────
  setLayer(id) {
    if (this.layerId === id) return;
    this.layerId = id;
    // An untouched view re-fits to the current canvas; one the user has framed
    // themselves is left exactly where they put it.
    if (this.view.fitted) this.fit();
    this.invalidate();
    if (this.hooks.onLayerChange) this.hooks.onLayerChange(this);
  }

  invalidate() {
    if (this._pending) return;
    this._pending = true;
    // rAF normally wins, but it is throttled to nothing while the document is
    // hidden and can be starved indefinitely when the window is merely
    // occluded. The timer guarantees the canvas still catches up.
    const run = () => {
      if (!this._pending) return;
      this._pending = false;
      cancelAnimationFrame(this._rafId);
      clearTimeout(this._timer);
      this.draw();
    };
    this._rafId = requestAnimationFrame(run);
    this._timer = setTimeout(run, 120);
  }

  _changed() {
    if (this.hooks.onViewChange) this.hooks.onViewChange(this);
    this.invalidate();
  }

  _observeSize() {
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(this.canvas.parentElement);
    this._resize();
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, Math.round(rect.width * this.dpr));
    this.h = Math.max(1, Math.round(rect.height * this.dpr));
    this.canvas.width = this.w;
    this.canvas.height = this.h;
    // A view the user has not framed themselves stays fitted across resizes -
    // including the first ResizeObserver callback, when the element finally has
    // a real size.
    if (this.view.fitted) this.fit(); else this.invalidate();
  }

  // ── interaction ────────────────────────────────────────────
  _bindEvents() {
    const cv = this.canvas;

    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const [px, py] = this._pointer(e);
      const [ux, uy] = this.toUnit(px, py);
      const view = this.view;
      const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018));
      const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, view.scale * factor));
      if (next === view.scale) return;
      view.scale = next;
      // keep the point under the cursor pinned
      view.cx = ux - (px - this.w / 2) / next;
      view.cy = uy - (py - this.h / 2) / next;
      view.fitted = false;
      this._changed();
    }, { passive: false });

    cv.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      try { cv.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
      const [px, py] = this._pointer(e);
      this._drag = { px, py, moved: 0, cx: this.view.cx, cy: this.view.cy };
      cv.classList.add('panning');
    });

    cv.addEventListener('pointermove', (e) => {
      const [px, py] = this._pointer(e);
      if (this._drag) {
        const dx = px - this._drag.px, dy = py - this._drag.py;
        this._drag.moved = Math.max(this._drag.moved, Math.hypot(dx, dy));
        const view = this.view;
        view.cx = this._drag.cx - dx / view.scale;
        view.cy = this._drag.cy - dy / view.scale;
        view.fitted = false;
        this._changed();
      } else {
        this._emitHover(px, py);
      }
    });

    const endDrag = (e) => {
      if (!this._drag) return;
      const wasClick = this._drag.moved < 3 * this.dpr;
      const [px, py] = this._pointer(e);
      this._drag = null;
      cv.classList.remove('panning');
      if (wasClick) {
        const idx = this._hitTest(px, py);
        if (idx != null && this.hooks.onPick) this.hooks.onPick(idx, this);
      }
    };
    cv.addEventListener('pointerup', endDrag);
    cv.addEventListener('pointercancel', endDrag);

    cv.addEventListener('pointerleave', () => {
      if (this.hooks.onHover) this.hooks.onHover(null, this);
    });

    cv.addEventListener('dblclick', (e) => { e.preventDefault(); this.fit(); });
    this.els.fit.addEventListener('click', () => this.fit());
  }

  _pointer(e) {
    const r = this.canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * this.dpr, (e.clientY - r.top) * this.dpr];
  }

  _emitHover(px, py) {
    if (!this.hooks.onHover) return;
    this.hooks.onHover(this._hitTest(px, py), this);
  }

  _hitTest(px, py) {
    const m = this.store.micrograph;
    if (!m || !m.ready) return null;
    const [u, v] = this.toUnit(px, py);

    if (this.space === 'field') {
      const r = this.sourceRect();
      const fu = (u - r.x) / r.w, fv = (v - r.y) / r.h;
      if (fu < 0 || fu >= 1 || fv < 0 || fv >= 1) return null;
      const col = Math.floor(fu * m.grid.cols);
      const row = Math.floor(fv * m.grid.rows);
      return row * m.grid.cols + col;
    }

    if (this.space === 'latent' && this.source && this.source.type === 'scatter') {
      const pts = this._latentPoints();
      const tol = 9 * this.dpr;
      let best = null, bestD = Infinity;
      for (let i = 0; i < pts.length; i++) {
        const [sx, sy] = this.toScreen(pts[i][0], pts[i][1]);
        const d = Math.hypot(sx - px, sy - py);
        if (d < tol && d < bestD) { bestD = d; best = i; }
      }
      return best;
    }
    return null;
  }

  _latentPoints() {
    const src = this.source;
    if (!src || src.type !== 'scatter') return [];
    if (!this.latentBounds || this.latentBounds.key !== this.sourceKey) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < src.xs.length; i++) {
        x0 = Math.min(x0, src.xs[i]); x1 = Math.max(x1, src.xs[i]);
        y0 = Math.min(y0, src.ys[i]); y1 = Math.max(y1, src.ys[i]);
      }
      const sx = (x1 - x0) || 1, sy = (y1 - y0) || 1;
      const span = Math.max(sx, sy) * 1.12;            // one isotropic scale
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const pts = src.xs.map((x, i) => [
        0.5 + (x - mx) / span,
        0.5 - (src.ys[i] - my) / span,               // y up, like a plot
      ]);
      this.latentBounds = { key: this.sourceKey, pts, span, mx, my };
    }
    return this.latentBounds.pts;
  }

  // ── drawing ────────────────────────────────────────────────
  draw() {
    const key = sourceKey(this.layerId, this.store);
    // A 'missing' source is usually just an image still in flight, so it is
    // never cached - the layer resolves itself on the redraw that img.onload
    // triggers.
    if (key !== this.sourceKey || !this.source || this.source.type === 'missing') {
      this.source = buildSource(this.layerId, this.store);
      this.sourceKey = key;
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = '#0b0d10';
    ctx.fillRect(0, 0, this.w, this.h);

    const src = this.source;
    const layer = LAYER_BY_ID[this.layerId] || {};
    this.els.label.textContent = layer.label || this.layerId;
    this.els.space.textContent = this.space;

    if (!src || src.type === 'missing') {
      this.els.empty.hidden = false;
      this.els.empty.textContent = (src && src.note) || 'no data';
      this.els.scalebar.replaceChildren();
      this.els.zoom.textContent = '—';
      this.root.title = (src && src.note) || '';
      return;
    }
    this.els.empty.hidden = true;

    if (src.type === 'raster') this._drawRaster(src);
    else if (src.type === 'scatter') this._drawScatter(src);

    this._drawHeader(src);
  }

  _drawRaster(src) {
    const ctx = this.ctx;
    const node = src.image || src.canvas;
    const r = this.sourceRect();
    const [x0, y0] = this.toScreen(r.x, r.y);
    const w = r.w * this.view.scale, h = r.h * this.view.scale;

    ctx.imageSmoothingEnabled = !this.store.opts.nearest;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(pickMip(node, w), x0, y0, w, h);
    ctx.imageSmoothingEnabled = true;

    if (this.space === 'field') this._drawFieldOverlay(r);
    else this._drawTileOverlay(r);
  }

  _drawFieldOverlay(rect) {
    const ctx = this.ctx;
    const m = this.store.micrograph;
    const st = this.store.state;
    const { rows, cols } = m.grid;
    const cw = (rect.w / cols) * this.view.scale;
    const ch = (rect.h / rows) * this.view.scale;
    const cellPx = Math.min(cw, ch);

    const cellRect = (idx) => {
      const r = Math.floor(idx / cols), c = idx % cols;
      const [x, y] = this.toScreen(rect.x + (c / cols) * rect.w, rect.y + (r / rows) * rect.h);
      return [x, y, cw, ch];
    };

    if (this.store.opts.grid && cellPx > 5) {
      ctx.save();
      ctx.strokeStyle = COLOR.grid;
      ctx.lineWidth = Math.max(1, 0.5 * this.dpr);
      const [gx0, gy0] = this.toScreen(rect.x, rect.y);
      const [gx1, gy1] = this.toScreen(rect.x + rect.w, rect.y + rect.h);
      ctx.beginPath();
      for (let c = 0; c <= cols; c++) {
        const x = gx0 + ((gx1 - gx0) * c) / cols;
        ctx.moveTo(x, gy0); ctx.lineTo(x, gy1);
      }
      for (let r = 0; r <= rows; r++) {
        const y = gy0 + ((gy1 - gy0) * r) / rows;
        ctx.moveTo(gx0, y); ctx.lineTo(gx1, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    if (st && this.store.opts.markers) {
      ctx.save();
      ctx.lineWidth = Math.max(1.4, 1.4 * this.dpr);
      for (const cap of st.captures) {
        const [x, y, w, h] = cellRect(cap.idx);
        ctx.strokeStyle = cap.kind === 'warmup' ? COLOR.warmup : COLOR.policy;
        ctx.globalAlpha = 0.95;
        ctx.strokeRect(x + ctx.lineWidth / 2, y + ctx.lineWidth / 2, w - ctx.lineWidth, h - ctx.lineWidth);
      }
      ctx.globalAlpha = 1;

      if (this.store.opts.stepNumbers && cellPx > 17 * this.dpr) {
        ctx.font = `${Math.max(9, Math.min(cellPx * 0.34, 16 * this.dpr))}px ui-monospace, monospace`;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        for (const cap of st.captures) {
          const [x, y] = cellRect(cap.idx);
          const label = cap.kind === 'warmup' ? 'W' : String(cap.step);
          const pad = 2 * this.dpr;
          ctx.fillStyle = 'rgba(0,0,0,0.62)';
          const tw = ctx.measureText(label).width;
          ctx.fillRect(x + pad, y + pad, tw + 2 * pad, ctx.font ? parseFloat(ctx.font) + pad : 12);
          ctx.fillStyle = cap.kind === 'warmup' ? COLOR.warmup : COLOR.policy;
          ctx.fillText(label, x + 2 * pad, y + 1.5 * pad);
        }
      }

      for (const idx of st.pending) {
        const [x, y, w, h] = cellRect(idx);
        ctx.save();
        ctx.strokeStyle = COLOR.pending;
        ctx.lineWidth = Math.max(2, 2 * this.dpr);
        ctx.setLineDash([5 * this.dpr, 3 * this.dpr]);
        ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
        ctx.setLineDash([]);
        ctx.strokeRect(x - 2 * this.dpr, y - 2 * this.dpr, w + 4 * this.dpr, h + 4 * this.dpr);
        ctx.restore();
      }
      ctx.restore();
    }

    const focus = this.store.focusIndex;
    if (focus != null && focus !== this.store.hoverIndex) {
      const [x, y, w, h] = cellRect(focus);
      ctx.save();
      ctx.strokeStyle = COLOR.focus;
      ctx.lineWidth = Math.max(1.5, 1.5 * this.dpr);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }

    const hover = this.store.hoverIndex;
    if (hover != null && hover >= 0 && hover < rows * cols) {
      const [x, y, w, h] = cellRect(hover);
      ctx.save();
      ctx.strokeStyle = COLOR.hover;
      ctx.lineWidth = Math.max(1.5, 1.5 * this.dpr);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
  }

  _drawTileOverlay(rect) {
    const ctx = this.ctx;
    const [x, y] = this.toScreen(rect.x, rect.y);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.lineWidth = this.dpr;
    ctx.strokeRect(x, y, rect.w * this.view.scale, rect.h * this.view.scale);
    ctx.restore();
  }

  _drawScatter(src) {
    const ctx = this.ctx;
    const pts = this._latentPoints();
    const st = this.store.state;
    const [lo, hi] = src.domain || [0, 1];
    const span = (hi - lo) || 1;
    const lut = src.cmap;
    const rBase = 3.1 * this.dpr;

    // axes through the data centroid
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = this.dpr;
    const [ax, ay] = this.toScreen(0.5, 0.5);
    ctx.beginPath();
    ctx.moveTo(0, ay); ctx.lineTo(this.w, ay);
    ctx.moveTo(ax, 0); ctx.lineTo(ax, this.h);
    ctx.stroke();
    ctx.restore();

    const capByIdx = new Map();
    if (st) for (const c of st.captures) capByIdx.set(c.idx, c);
    const pending = new Set(st ? st.pending : []);

    for (let i = 0; i < pts.length; i++) {
      const [sx, sy] = this.toScreen(pts[i][0], pts[i][1]);
      if (sx < -20 || sy < -20 || sx > this.w + 20 || sy > this.h + 20) continue;
      const t = (src.values[i] - lo) / span;
      ctx.beginPath();
      ctx.arc(sx, sy, rBase, 0, Math.PI * 2);
      ctx.fillStyle = cssFromLut(lut, t);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      ctx.lineWidth = this.dpr * 0.8;
      ctx.stroke();
    }

    // captured points on top, ringed and labelled
    ctx.save();
    ctx.font = `${10 * this.dpr}px ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    for (const [idx, cap] of capByIdx) {
      if (idx >= pts.length) continue;
      const [sx, sy] = this.toScreen(pts[idx][0], pts[idx][1]);
      ctx.beginPath();
      ctx.arc(sx, sy, rBase + 2.6 * this.dpr, 0, Math.PI * 2);
      ctx.strokeStyle = cap.kind === 'warmup' ? COLOR.warmup : COLOR.policy;
      ctx.lineWidth = 1.8 * this.dpr;
      ctx.stroke();
      if (this.store.opts.stepNumbers) {
        ctx.fillStyle = cap.kind === 'warmup' ? COLOR.warmup : COLOR.policy;
        ctx.fillText(cap.kind === 'warmup' ? 'W' : String(cap.step), sx, sy - 5.5 * this.dpr);
      }
    }
    for (const idx of pending) {
      if (idx >= pts.length) continue;
      const [sx, sy] = this.toScreen(pts[idx][0], pts[idx][1]);
      ctx.beginPath();
      ctx.arc(sx, sy, rBase + 5 * this.dpr, 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.pending;
      ctx.lineWidth = 2 * this.dpr;
      ctx.setLineDash([4 * this.dpr, 3 * this.dpr]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    const hover = this.store.hoverIndex;
    if (hover != null && hover < pts.length) {
      const [sx, sy] = this.toScreen(pts[hover][0], pts[hover][1]);
      ctx.save();
      ctx.beginPath();
      ctx.arc(sx, sy, rBase + 7 * this.dpr, 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.hover;
      ctx.lineWidth = 1.6 * this.dpr;
      ctx.stroke();
      ctx.restore();
    }

    const focus = this.store.focusIndex;
    if (focus != null && focus < pts.length && focus !== hover) {
      const [sx, sy] = this.toScreen(pts[focus][0], pts[focus][1]);
      ctx.save();
      ctx.beginPath();
      ctx.arc(sx, sy, rBase + 7 * this.dpr, 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.focus;
      ctx.lineWidth = 1.4 * this.dpr;
      ctx.stroke();
      ctx.restore();
    }
  }

  _drawHeader(src) {
    const bar = this.els.scalebar;
    bar.replaceChildren();
    if (src.domain) {
      const [lo, hi] = src.domain;
      const fmt = (v) => (Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0)
        ? v.toExponential(1) : v.toFixed(3));
      const left = document.createElement('span');
      left.textContent = fmt(lo);
      const right = document.createElement('span');
      right.textContent = fmt(hi);
      bar.append(left, colorbarCanvas(src.cmap || this.store.opts.cmap, 54, 8), right);
    } else if (src.note) {
      const s = document.createElement('span');
      s.textContent = src.note;
      bar.append(s);
    }
    this.root.title = src.note || '';

    const m = this.store.micrograph;
    if (this.space === 'field' && m && m.ready) {
      const perPatch = (this.view.scale / m.grid.cols) / this.dpr;
      this.els.zoom.textContent = `${perPatch.toFixed(1)} px/patch`;
    } else {
      const rel = this.view.scale / (Math.min(this.w, this.h) * PAD);
      this.els.zoom.textContent = `${rel.toFixed(2)}×`;
    }
  }
}

// Inline LUT lookup — the scatter loop runs once per patch per frame.
function cssFromLut(name, t) {
  const lut = cmapTable(name);
  const i = Math.max(0, Math.min(255, Math.round((Number.isFinite(t) ? t : 0) * 255))) * 3;
  return `rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]})`;
}
