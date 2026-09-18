"""Run the same synthetic AutoRASOR demo pipeline without a browser or plots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autorasor.io import save_run  # noqa: E402
from demo_core import DemoSession, EngineConfig, RunConfig  # noqa: E402
from pipeline import MicrographParams, Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AutoRASOR synthetic acquisition and save raw evaluation data")
    parser.add_argument("--output", required=True, help="New or existing output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--csv", default=None, help="OPMD metadata CSV override")
    parser.add_argument("--image-dir", default=None, help="OPMD phase-field image directory override")
    args = parser.parse_args()

    try:
        pipeline = Pipeline(
            csv_path=args.csv,
            image_dir=args.image_dir,
            device=args.device,
            dino_layer=5,
            use_tta=not args.no_tta,
        )
        session = DemoSession(pipeline, EngineConfig(
            batch_size=args.batch_size,
            lfps_warmup_count=args.warmup,
            capture_budget=args.budget,
            seed=args.seed,
        ))
        run_config = RunConfig(
            policy="ambiguity",
            budget=args.budget,
            batch_size=args.batch_size,
            acq="qlognei",
            warmup_seeds=args.warmup,
            seed=args.seed,
        )
        session.run_cfg = run_config
        params = MicrographParams(seed=args.seed)
        session.new_micrograph(params, keep_gpr=False)
        session.run_to_budget()
    except (FileNotFoundError, OSError) as error:
        raise SystemExit(
            f"AutoRASOR assets are unavailable: {error}\n"
            "Place the OPMD CSV/images as described in data/README.md or pass --csv and --image-dir."
        ) from error

    final_state = session.history[-1]
    result = {
        "config": {
            "engine": asdict(session.engine_cfg),
            "run": asdict(session.run_cfg),
            "micrograph": asdict(params),
            "device": pipeline.device,
            "dino_layer": 5,
            "use_tta": not args.no_tta,
        },
        "observations": [asdict(capture) for capture in final_state.captures],
        "selections": [
            {"step": state.step, "pending_indices": state.pending}
            for state in session.history
            if state.pending
        ],
        "metrics": [
            {"step": state.step, "n_carried": state.n_carried, "fit_seconds": state.fit_seconds, **state.metrics}
            for state in session.history
        ],
    }
    output = save_run(Path(args.output), result)
    print(f"Saved AutoRASOR run to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
