# AutoRASOR - Autonomous Rapid SEM Operator

Full code and datasets will be uploaded soon, current repo has a web demo of the AutoRASOR pipeline.

## Repository layout

| Path | Purpose |
|---|---|
| `AutoRASOR_web_demo/` | Interactive synthetic simulation and headless evaluation |
| `autorasor/` | Configuration, ambiguity/active learning/LFPS setup and code, DINOv3 feature extraction, and synthetic data generator |
| `data/` | dataset shown in the paper and link to OPMD dataset |
| `example_pipeline.py` | AutoRASOR pipeline |

## Installation

```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

DINOv3 weights are governed by Meta's license and are not redistributed here. The synthetic demo requires the OPMD metadata CSV and phase-field images described in `data/README.md`. Follow the instructions there to download and extract them before running the demo.

## Interactive Simulation

The local web app visualizes the low- and high-magnification fields, LFPS warmup, learned ambiguity, ground-truth ambiguity, acquisition scores, and stepwise evaluation. 

```bash
python AutoRASOR_web_demo/run_demo.py
```

It supports LFPS, the full ambiguity-driven active learning campaign, and Random sampling as a control.


## License
MIT