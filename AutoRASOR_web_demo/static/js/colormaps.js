// Perceptual colormaps sampled from matplotlib, interpolated in sRGB.

const STOPS = {
  viridis: ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E',
            '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
  inferno: ['#000004', '#1B0C41', '#4A0C6B', '#781C6D', '#A52C60',
            '#CF4446', '#ED6925', '#FB9A06', '#F7D13D', '#FCFFA4'],
  magma:   ['#000004', '#180F3D', '#440F76', '#721F81', '#9E2F7F',
            '#CD4071', '#F1605D', '#FD9668', '#FEC98D', '#FCFDBF'],
  plasma:  ['#0D0887', '#47039F', '#7301A8', '#9C179E', '#BD3786',
            '#D8576B', '#ED7953', '#FA9E3B', '#FDC926', '#F0F921'],
  turbo:   ['#30123B', '#4145AB', '#4675ED', '#39A2FC', '#1BCFD4',
            '#24ECA6', '#61FC6C', '#A4FC3B', '#D1E834', '#F3C63A',
            '#FE9B2D', '#F36315', '#D93806', '#B11901', '#7A0403'],
  gray:    ['#000000', '#FFFFFF'],
  // Diverging, for signed error maps: blue = under-predicted, red = over.
  rdbu:    ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
            '#FDDBC7', '#F4A582', '#D6604D', '#B2182B', '#67001F'],
};

const TABLES = {};

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function buildTable(name) {
  const stops = STOPS[name].map(hexToRgb);
  const size = 256;
  const table = new Uint8Array(size * 3);
  for (let i = 0; i < size; i++) {
    const t = (i / (size - 1)) * (stops.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, stops.length - 1);
    const f = t - lo;
    for (let c = 0; c < 3; c++) {
      table[i * 3 + c] = Math.round(stops[lo][c] * (1 - f) + stops[hi][c] * f);
    }
  }
  return table;
}

export function table(name) {
  const key = STOPS[name] ? name : 'viridis';
  if (!TABLES[key]) TABLES[key] = buildTable(key);
  return TABLES[key];
}

/** Sample a colormap. `t` is clamped to [0,1]. Returns [r,g,b] 0-255. */
export function sample(name, t) {
  const lut = table(name);
  const i = Math.max(0, Math.min(255, Math.round((Number.isFinite(t) ? t : 0) * 255)));
  return [lut[i * 3], lut[i * 3 + 1], lut[i * 3 + 2]];
}

export function cssColor(name, t) {
  const [r, g, b] = sample(name, t);
  return `rgb(${r},${g},${b})`;
}

/** Horizontal colorbar strip, for the viewport headers. */
export function colorbarCanvas(name, w = 56, h = 8) {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(w, h);
  const lut = table(name);
  for (let x = 0; x < w; x++) {
    const i = Math.round((x / Math.max(w - 1, 1)) * 255) * 3;
    for (let y = 0; y < h; y++) {
      const o = (y * w + x) * 4;
      img.data[o] = lut[i];
      img.data[o + 1] = lut[i + 1];
      img.data[o + 2] = lut[i + 2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}
