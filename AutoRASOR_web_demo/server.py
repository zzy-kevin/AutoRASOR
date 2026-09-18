"""
Stdlib HTTP server for the AutoRASOR web demo.
==============================================

No web framework on purpose: the project venv has torch/BoTorch/DINOv3 but no
Flask, and a local single-user demo does not need one. `ThreadingHTTPServer` +
a session lock is enough.

Long jobs (building a micrograph: generate 196 tiles, two DINOv3 passes, GT
ambiguity) run on a worker thread so the browser can poll /api/progress and draw
a progress bar instead of hanging on a 15-second request.
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from demo_core import DemoSession, RunConfig
from pipeline import MicrographParams

STATIC_DIR = Path(__file__).resolve().parent / 'static'
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
}


# ============================================================
# JSON helpers
# ============================================================

def _clean(value: Any) -> Any:
    """numpy -> json, with NaN/inf flattened to null (JSON has no NaN)."""
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


# ============================================================
# Application
# ============================================================

class DemoApp:
    """Routes + serialization. Owns the session lock."""

    def __init__(self, session: DemoSession, default_params: MicrographParams):
        self.session = session
        self.default_params = default_params
        self.lock = threading.Lock()
        self.build_error: Optional[str] = None
        self._worker: Optional[threading.Thread] = None

    # -- serialization ---------------------------------------------------
    def micrograph_payload(self) -> Dict[str, Any]:
        s = self.session
        m = s.micrograph
        if m is None:
            return {'ready': False}
        rows, cols = m.grid_shape
        pca_low = s.embedding_pca('low')
        pca_high = s.embedding_pca('high')
        return {
            'ready': True,
            'serial': s.micrograph_serial,
            'seed': m.params.seed,
            'grid': {'rows': rows, 'cols': cols},
            'n_points': m.n_points,
            'low_mag_size': list(m.low_mag.shape),
            'tile_size_px': int(m.tiles.shape[1]),
            'build_seconds': round(m.build_seconds, 2),
            'gt_ambiguity': _clean(m.gt_ambiguity),
            'af_gt': _clean(m.af_gt.ravel()),
            'mob_gt': _clean(m.mob_gt.ravel()),
            'pca_low': {
                'scores': _clean(pca_low['scores']),
                'evr': _clean(pca_low['explained_variance_ratio']),
            },
            'pca_high': {
                'scores': _clean(pca_high['scores']),
                'evr': _clean(pca_high['explained_variance_ratio']),
            },
            'params': {
                'seed': m.params.seed,
                'local_blur_max_sigma': m.params.local_blur_max_sigma,
                'global_blur_sigma': m.params.global_blur_sigma,
                'gaussian_noise_std': m.params.gaussian_noise_std,
                'shot_noise_scale': m.params.shot_noise_scale,
                'heatmap_scale_af': m.params.heatmap_scale_af,
                'heatmap_scale_mob': m.params.heatmap_scale_mob,
            },
        }

    def state_payload(self) -> Dict[str, Any]:
        s = self.session
        if s.micrograph is None or not s.history:
            return {'ready': False}
        st = s.state
        cfg = s.run_cfg
        return {
            'ready': True,
            'serial': s.micrograph_serial,
            'cursor': s.cursor,
            'step': st.step,
            'n_states': len(s.history),
            'frontier_step': s.history[-1].step,
            'at_frontier': s.cursor == len(s.history) - 1,
            'can_advance': bool(s.cursor < len(s.history) - 1 or s.history[-1].pending),
            'run': {
                'policy': cfg.policy,
                'budget': cfg.budget,
                'batch_size': cfg.batch_size,
                'acq': cfg.acq,
                'beta': cfg.beta,
                'warmup_seeds': cfg.warmup_seeds,
                'seed': cfg.seed,
            },
            'engine': {
                'kernel': s.engine_cfg.kernel,
                'fit_mode': s.engine_cfg.fit_mode,
                'noise_mode': s.engine_cfg.noise_mode,
                'metric': s.engine_cfg.metric,
                'k_neighbors': s.engine_cfg.k_neighbors_ambiguity,
                'normalize_features': s.engine_cfg.normalize_features,
                'n_observed': s.engine.n_observed if s.engine else 0,
            },
            'captures': [{'idx': c.idx, 'step': c.step, 'kind': c.kind} for c in st.captures],
            'pending': st.pending,
            'gp': {
                'active': st.gp_active,
                'flat': st.gp_flat,
                'mean': _clean(st.gp_mean) if st.gp_mean is not None else None,
                'std': _clean(st.gp_std) if st.gp_std is not None else None,
                'acq': _clean(st.gp_acq) if st.gp_acq is not None else None,
            },
            'obs_ambiguity': _clean(st.obs_ambiguity),
            'metrics': _clean(st.metrics),
            'n_carried': st.n_carried,
            'fit_seconds': round(st.fit_seconds, 3),
        }

    def progress_payload(self) -> Dict[str, Any]:
        msg, frac = self.session.progress
        return {
            'status': self.session.status,
            'message': msg,
            'frac': round(float(frac), 3),
            'serial': self.session.micrograph_serial,
            'error': self.build_error,
        }

    # -- actions ---------------------------------------------------------
    def start_micrograph(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self.session.status == 'building':
            return {'started': False, 'reason': 'a build is already running'}
        base = self.session.micrograph.params if self.session.micrograph else self.default_params
        params = MicrographParams(
            seed=int(body.get('seed', base.seed)),
            grid_rows=base.grid_rows,
            grid_cols=base.grid_cols,
            tile_size=base.tile_size,
            high_mag_size=base.high_mag_size,
            heatmap_scale_af=float(body.get('heatmap_scale_af', base.heatmap_scale_af)),
            heatmap_scale_mob=float(body.get('heatmap_scale_mob', base.heatmap_scale_mob)),
            local_blur_max_sigma=float(body.get('local_blur_max_sigma', base.local_blur_max_sigma)),
            global_blur_sigma=float(body.get('global_blur_sigma', base.global_blur_sigma)),
            gaussian_noise_std=float(body.get('gaussian_noise_std', base.gaussian_noise_std)),
            shot_noise_scale=float(body.get('shot_noise_scale', base.shot_noise_scale)),
            allow_duplicates=base.allow_duplicates,
        )
        keep = bool(body.get('keep_gpr', True))
        self.build_error = None

        def work() -> None:
            try:
                with self.lock:
                    self.session.new_micrograph(params, keep_gpr=keep)
            except Exception:
                self.build_error = traceback.format_exc(limit=3)
                self.session.status = 'idle'
                print('[demo] micrograph build failed:\n' + self.build_error)

        self._worker = threading.Thread(target=work, daemon=True)
        self.session.status = 'building'
        self.session.progress = ('Starting', 0.0)
        self._worker.start()
        return {'started': True}

    def start_run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        cfg = RunConfig(
            policy=str(body.get('policy', 'ambiguity')),
            budget=int(body.get('budget', 40)),
            batch_size=int(body.get('batch_size', 2)),
            acq='qlognei',
            beta=float(body.get('beta', 1.0)),
            warmup_seeds=int(body.get('warmup_seeds', 8)),
            seed=int(body.get('seed', 42)),
        )
        self.session.start_run(cfg)
        return self.state_payload()

    def do_step(self, body: Dict[str, Any]) -> Dict[str, Any]:
        direction = str(body.get('dir', 'next'))
        if direction == 'next':
            self.session.step_forward()
        elif direction == 'prev':
            self.session.step_back()
        elif direction == 'goto':
            self.session.goto(int(body.get('step', 0)))
        elif direction == 'end':
            self.session.run_to_budget(max_steps=int(body.get('max_steps', 400)))
        elif direction == 'start':
            self.session.cursor = 0
        return self.state_payload()

    # -- routing ---------------------------------------------------------
    def handle(self, method: str, path: str, query: Dict[str, list],
               body: Dict[str, Any]) -> Tuple[int, str, bytes]:
        s = self.session

        if method == 'GET':
            if path in ('/', '/index.html'):
                return self._static('index.html')
            if path.startswith('/static/'):
                return self._static(path[len('/static/'):])
            if path == '/api/progress':
                return self._json(self.progress_payload())
            # Reads during a build would either race the worker or block on its
            # lock for ~15s; report "busy" instead and let the client keep polling.
            if path == '/api/micrograph':
                if s.status == 'building':
                    return self._json({'ready': False, 'building': True})
                return self._json(self.micrograph_payload())
            if path == '/api/state':
                if s.status == 'building':
                    return self._json({'ready': False, 'building': True})
                return self._json(self.state_payload())
            if path == '/api/bootstrap':
                return self._json({
                    'device': self.session.pipeline.device,
                    'dino_layer': self.session.pipeline.dino_layer,
                    'use_tta': self.session.pipeline.use_tta,
                    'defaults': {
                        'seed': self.default_params.seed,
                        'local_blur_max_sigma': self.default_params.local_blur_max_sigma,
                        'global_blur_sigma': self.default_params.global_blur_sigma,
                        'gaussian_noise_std': self.default_params.gaussian_noise_std,
                        'shot_noise_scale': self.default_params.shot_noise_scale,
                        'heatmap_scale_af': self.default_params.heatmap_scale_af,
                        'heatmap_scale_mob': self.default_params.heatmap_scale_mob,
                    },
                    'building': s.status == 'building',
                    'micrograph': {'ready': False} if s.status == 'building' else self.micrograph_payload(),
                    'state': {'ready': False} if s.status == 'building' else self.state_payload(),
                })
            if path.startswith('/img/'):
                return self._image(path[len('/img/'):])

        if method == 'POST':
            if path == '/api/micrograph':
                return self._json(self.start_micrograph(body))
            with self.lock:
                if s.micrograph is None:
                    return self._json({'error': 'no micrograph loaded'}, status=409)
                if path == '/api/run':
                    return self._json(self.start_run(body))
                if path == '/api/step':
                    return self._json(self.do_step(body))
                if path == '/api/reset_gpr':
                    s.reset_gpr()
                    return self._json(self.state_payload())

        return 404, 'text/plain; charset=utf-8', b'not found'

    # -- responders ------------------------------------------------------
    def _json(self, payload: Any, status: int = 200) -> Tuple[int, str, bytes]:
        return status, 'application/json', json.dumps(payload).encode('utf-8')

    def _static(self, rel: str) -> Tuple[int, str, bytes]:
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return 404, 'text/plain; charset=utf-8', b'not found'
        ctype = CONTENT_TYPES.get(target.suffix.lower(), 'application/octet-stream')
        return 200, ctype, target.read_bytes()

    def _image(self, rel: str) -> Tuple[int, str, bytes]:
        m = self.session.micrograph
        if m is None:
            return 409, 'text/plain; charset=utf-8', b'no micrograph'
        name = rel.split('?')[0]
        try:
            if name.startswith('tile/'):
                idx = int(Path(name).stem)
                if not (0 <= idx < m.n_points):
                    return 404, 'text/plain; charset=utf-8', b'tile out of range'
                return 200, 'image/png', m.tile_png(idx)
            key = Path(name).stem
            return 200, 'image/png', m.png(key)
        except (KeyError, ValueError):
            return 404, 'text/plain; charset=utf-8', b'unknown image'


# ============================================================
# HTTP plumbing
# ============================================================

def make_handler(app: DemoApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'AutoRASORDemo/1.0'
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt: str, *args: Any) -> None:
            if args and '/api/progress' in str(args[0]):
                return
            print('[demo] %s' % (fmt % args))

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            body: Dict[str, Any] = {}
            if method == 'POST':
                length = int(self.headers.get('Content-Length') or 0)
                if length:
                    try:
                        body = json.loads(self.rfile.read(length).decode('utf-8'))
                    except json.JSONDecodeError:
                        body = {}
            try:
                status, ctype, payload = app.handle(
                    method, parsed.path, parse_qs(parsed.query), body
                )
            except Exception:
                trace = traceback.format_exc()
                print('[demo] request failed:\n' + trace)
                status, ctype = 500, 'application/json'
                payload = json.dumps({'error': trace.splitlines()[-1]}).encode('utf-8')

            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(payload)))
            if ctype == 'image/png':
                self.send_header('Cache-Control', 'public, max-age=86400')
            else:
                self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:      # noqa: N802
            self._dispatch('GET')

        def do_POST(self) -> None:     # noqa: N802
            self._dispatch('POST')

    return Handler


def serve(app: DemoApp, host: str = '127.0.0.1', port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd
