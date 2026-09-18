"""
AutoRASOR interactive demo - entry point.
========================================

    .venv/Scripts/python AutoRASOR_web_demo/run_demo.py

Loads DINOv3 once, starts a local HTTP server, and opens the browser. The first
micrograph is built on demand from the page (so you see the progress bar).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_core import DemoSession, EngineConfig, RunConfig  # noqa: E402
from pipeline import MicrographParams, Pipeline  # noqa: E402
from server import DemoApp, serve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description='AutoRASOR LFPS / ambiguity active-learning demo')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--device', default='cuda', help="'cuda' or 'cpu'")
    ap.add_argument('--dino-layer', type=int, default=5, help='DINOv3 block to read (canonical: 5)')
    ap.add_argument('--no-tta', action='store_true', help='disable D4 test-time augmentation (faster, less canonical)')
    ap.add_argument('--seed', type=int, default=7, help='initial micrograph seed')
    ap.add_argument('--csv', default=None, help='override OPMD metadata CSV')
    ap.add_argument('--image-dir', default=None, help='override phase-field image directory')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--prebuild', action='store_true', help='build the first micrograph before serving')
    args = ap.parse_args()

    print('[demo] loading pipeline (DINOv3 + phase-field generator)...')
    t0 = time.time()
    pipeline = Pipeline(
        csv_path=args.csv,
        image_dir=args.image_dir,
        device=args.device,
        dino_layer=args.dino_layer,
        use_tta=not args.no_tta,
    )
    print('[demo] pipeline ready in %.1fs (device=%s, layer=%d, tta=%s)'
          % (time.time() - t0, pipeline.device, args.dino_layer, not args.no_tta))

    session = DemoSession(pipeline, EngineConfig())
    session.run_cfg = RunConfig()
    params = MicrographParams(seed=args.seed)
    app = DemoApp(session, params)

    if args.prebuild:
        print('[demo] building first micrograph...')
        session.new_micrograph(params, keep_gpr=False)
        print('[demo] micrograph ready in %.1fs' % session.micrograph.build_seconds)

    httpd = serve(app, host=args.host, port=args.port)
    url = 'http://%s:%d/' % (args.host, args.port)
    print('[demo] serving at %s   (Ctrl+C to stop)' % url)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n[demo] shutting down')
        httpd.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
