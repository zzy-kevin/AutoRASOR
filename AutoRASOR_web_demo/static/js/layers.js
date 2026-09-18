// Layer registry: turns backend arrays into things a viewport can draw.
//
// Three coordinate spaces:
//   field   the whole 14x14 field of view, normalized to the unit square
//   tile    one high-magnification tile (also the unit square)
//   latent  DINOv3 PC1/PC2, affinely mapped into the unit square
// Two viewports sync their pan/zoom whenever they are in the same space.

import { sample } from './colormaps.js';

export const LAYERS = [
  { id: 'lowmag',       label: 'Low-mag micrograph',        space: 'field',  group: 'Micrograph' },
  { id: 'lowmag_clean', label: 'Low-mag, undegraded',       space: 'field',  group: 'Micrograph' },
  { id: 'highmag',      label: 'High-mag montage',          space: 'field',  group: 'Micrograph' },
  { id: 'tile',         label: 'Selected high-mag tile',    space: 'tile',   group: 'Micrograph' },
  { id: 'tile_low',     label: 'Selected low-mag patch',    space: 'tile',   group: 'Micrograph' },
  { id: 'pca_low',      label: 'DINOv3 PCA RGB — low-mag',  space: 'field',  group: 'Embedding' },
  { id: 'pca_high',     label: 'DINOv3 PCA RGB — high-mag', space: 'field',  group: 'Embedding' },
  { id: 'latent_low',   label: 'Latent PC1/PC2 — low-mag',  space: 'latent', group: 'Embedding' },
  { id: 'latent_high',  label: 'Latent PC1/PC2 — high-mag', space: 'latent', group: 'Embedding' },
  { id: 'gpr_mean',     label: 'Learned ambiguity (GPR)',   space: 'field',  group: 'Ambiguity' },
  { id: 'gt_amb',       label: 'Ground-truth ambiguity',    space: 'field',  group: 'Ambiguity' },
  { id: 'gpr_std',      label: 'GPR posterior σ',           space: 'field',  group: 'Ambiguity' },
  { id: 'acq',          label: 'Acquisition score',         space: 'field',  group: 'Ambiguity' },
  { id: 'err',          label: 'GPR − ground truth',        space: 'field',  group: 'Ambiguity' },
  { id: 'af_gt',        label: 'GT area fraction',          space: 'field',  group: 'Physics' },
  { id: 'mob_gt',       label: 'GT mobility',               space: 'field',  group: 'Physics' },
];

export const LAYER_BY_ID = Object.fromEntries(LAYERS.map((l) => [l.id, l]));

const IMAGE_LAYERS = { lowmag: 'lowmag', lowmag_clean: 'lowmag_clean', highmag: 'highmag' };

// ── helpers ──────────────────────────────────────────────────

function finite(arr) {
  const out = [];
  for (const v of arr || []) if (Number.isFinite(v)) out.push(v);
  return out;
}

function extent(arr) {
  const f = finite(arr);
  if (!f.length) return [0, 1];
  let lo = Infinity, hi = -Infinity;
  for (const v of f) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-12) { hi = lo + 1e-12; }
  return [lo, hi];
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Colormapped rows x cols canvas, one pixel per patch. */
function gridCanvas(values, rows, cols, cmap, domain) {
  const cv = document.createElement('canvas');
  cv.width = cols; cv.height = rows;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  for (let i = 0; i < rows * cols; i++) {
    const v = values[i];
    const o = i * 4;
    if (!Number.isFinite(v)) {
      img.data[o] = 40; img.data[o + 1] = 44; img.data[o + 2] = 52; img.data[o + 3] = 255;
      continue;
    }
    const [r, g, b] = sample(cmap, (v - lo) / span);
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

/**
 * PC1-3 -> RGB. Each component is robustly rescaled to [-1,1] and then damped by
 * its explained-variance ratio relative to PC1, so PC2/PC3 cannot shout over PC1.
 */
function pcaCanvas(scores, evr, rows, cols, weighted) {
  const n = rows * cols;
  const cv = document.createElement('canvas');
  cv.width = cols; cv.height = rows;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(cols, rows);

  const scales = [];
  const weights = [];
  for (let c = 0; c < 3; c++) {
    const col = [];
    for (let i = 0; i < n; i++) col.push(scores[i][c]);
    col.sort((a, b) => a - b);
    const lo = quantile(col, 0.02), hi = quantile(col, 0.98);
    const mid = (lo + hi) / 2;
    const half = Math.max((hi - lo) / 2, 1e-9);
    scales.push([mid, half]);
    weights.push(weighted ? (evr[c] / (evr[0] || 1)) : 1);
  }

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    for (let c = 0; c < 3; c++) {
      const [mid, half] = scales[c];
      let v = (scores[i][c] - mid) / half;          // ~[-1, 1]
      v = Math.max(-1, Math.min(1, v)) * weights[c];
      img.data[o + c] = Math.round(255 * (0.5 + 0.5 * v));
    }
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return { canvas: cv, weights };
}

function tilePatchCanvas(store, index) {
  const img = store.images.lowmag;
  if (!img || !img.complete) return null;
  const { rows, cols } = store.micrograph.grid;
  const size = Math.round(img.naturalWidth / cols);
  const r = Math.floor(index / cols), c = index % cols;
  const cv = document.createElement('canvas');
  cv.width = size; cv.height = size;
  const ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, c * size, r * size, size, size, 0, 0, size, size);
  return cv;
}

// ── source construction ──────────────────────────────────────

function gpArray(store, key) {
  const gp = store.state && store.state.gp;
  return gp && gp.active ? gp[key] : null;
}

function ambiguityDomain(store, values) {
  // Ground truth and the GP's learned map share one scale so the two viewports
  // are directly comparable side by side.
  if (!store.opts.sharedScale) return extent(values);
  const gt = store.micrograph.gt_ambiguity;
  const mean = gpArray(store, 'mean');
  const pool = finite(gt).concat(mean ? finite(mean) : []);
  return pool.length ? extent(pool) : extent(values);
}

/**
 * Build a drawable source for a layer. Returns one of:
 *   {type:'raster',  canvas|image, domain, cmap, unit, note}
 *   {type:'scatter', xs, ys, values, domain, cmap, axis, note}
 *   {type:'missing', note}
 */
export function buildSource(id, store) {
  const m = store.micrograph;
  if (!m || !m.ready) return { type: 'missing', note: 'no micrograph' };
  const { rows, cols } = m.grid;
  const cmap = store.opts.cmap;

  if (IMAGE_LAYERS[id]) {
    const image = store.images[IMAGE_LAYERS[id]];
    if (!image || !image.complete || !image.naturalWidth) {
      return { type: 'missing', note: 'loading image…' };
    }
    return { type: 'raster', image, note: `${image.naturalWidth}×${image.naturalHeight} px` };
  }

  if (id === 'tile' || id === 'tile_low') {
    const idx = store.focusIndex;
    if (idx == null) return { type: 'missing', note: 'no tile selected' };
    const r = Math.floor(idx / cols), c = idx % cols;
    if (id === 'tile_low') {
      const canvas = tilePatchCanvas(store, idx);
      if (!canvas) return { type: 'missing', note: 'loading…' };
      return { type: 'raster', canvas, note: `patch ${idx} (r${r} c${c}) · ${canvas.width}² px` };
    }
    const image = store.tileImage(idx);
    if (!image || !image.complete || !image.naturalWidth) {
      return { type: 'missing', note: 'loading tile…' };
    }
    return { type: 'raster', image, note: `tile ${idx} (r${r} c${c}) · ${image.naturalWidth}² px` };
  }

  if (id === 'pca_low' || id === 'pca_high') {
    const pca = id === 'pca_low' ? m.pca_low : m.pca_high;
    const { canvas, weights } = pcaCanvas(
      pca.scores, pca.evr, rows, cols, store.opts.evrWeighted,
    );
    const evrTxt = pca.evr.map((v) => (100 * v).toFixed(1) + '%').join(' / ');
    const wTxt = weights.map((v) => v.toFixed(2)).join(' / ');
    return {
      type: 'raster', canvas, rgb: true,
      note: `PC1-3 = R/G/B · var ${evrTxt} · weights ${wTxt}`,
    };
  }

  if (id === 'latent_low' || id === 'latent_high') {
    const pca = id === 'latent_low' ? m.pca_low : m.pca_high;
    const xs = pca.scores.map((s) => s[0]);
    const ys = pca.scores.map((s) => s[1]);
    const colorBy = store.opts.latentColor;
    let values = m.gt_ambiguity;
    let domain = null;
    if (colorBy === 'gpr' && gpArray(store, 'mean')) values = gpArray(store, 'mean');
    domain = ambiguityDomain(store, values);
    const evrTxt = `PC1 ${(100 * pca.evr[0]).toFixed(1)}% · PC2 ${(100 * pca.evr[1]).toFixed(1)}%`;
    return {
      type: 'scatter', xs, ys, values, domain, cmap,
      axis: { x: 'PC1', y: 'PC2' },
      note: `${xs.length} patches · ${evrTxt}`,
    };
  }

  if (id === 'gt_amb') {
    const values = m.gt_ambiguity;
    const domain = ambiguityDomain(store, values);
    return {
      type: 'raster', canvas: gridCanvas(values, rows, cols, cmap, domain),
      domain, cmap, values, note: 'weighted variance of neighbour high-mag embeddings',
    };
  }

  if (id === 'gpr_mean') {
    const values = gpArray(store, 'mean');
    if (!values) return { type: 'missing', note: 'GP not fitted yet (needs ≥5 captures)' };
    const domain = ambiguityDomain(store, values);
    return {
      type: 'raster', canvas: gridCanvas(values, rows, cols, cmap, domain),
      domain, cmap, values,
      note: store.state.gp.flat ? 'posterior is still flat — more captures needed' : 'GP posterior mean',
    };
  }

  if (id === 'gpr_std' || id === 'acq') {
    const values = gpArray(store, id === 'gpr_std' ? 'std' : 'acq');
    if (!values) return { type: 'missing', note: 'GP not fitted yet (needs ≥5 captures)' };
    const domain = extent(values);
    return {
      type: 'raster', canvas: gridCanvas(values, rows, cols, cmap, domain),
      domain, cmap, values,
      note: id === 'gpr_std' ? 'posterior standard deviation' : 'mean + β·σ',
    };
  }

  if (id === 'err') {
    const mean = gpArray(store, 'mean');
    if (!mean) return { type: 'missing', note: 'GP not fitted yet (needs ≥5 captures)' };
    const gt = m.gt_ambiguity;
    const values = mean.map((v, i) => (Number.isFinite(v) && Number.isFinite(gt[i]) ? v - gt[i] : NaN));
    const a = Math.max(...finite(values).map(Math.abs), 1e-9);
    const domain = [-a, a];
    return {
      type: 'raster', canvas: gridCanvas(values, rows, cols, 'rdbu', domain),
      domain, cmap: 'rdbu', values, note: 'red = GP over-predicts, blue = under-predicts',
    };
  }

  if (id === 'af_gt' || id === 'mob_gt') {
    const values = id === 'af_gt' ? m.af_gt : m.mob_gt;
    const domain = extent(values);
    return {
      type: 'raster', canvas: gridCanvas(values, rows, cols, cmap, domain),
      domain, cmap, values,
      note: id === 'af_gt' ? 'phase-field area fraction placed per tile' : 'phase-field mobility placed per tile',
    };
  }

  return { type: 'missing', note: `unknown layer ${id}` };
}

/** Cache key: rebuild a source only when something it depends on changed. */
export function sourceKey(id, store) {
  const o = store.opts;
  const stepKey = store.state ? `${store.state.serial}:${store.state.step}:${store.state.cursor}` : 'nostate';
  return [
    id, store.serial, stepKey, o.cmap, o.sharedScale, o.evrWeighted, o.latentColor,
    (id === 'tile' || id === 'tile_low') ? store.focusIndex : '',
  ].join('|');
}
