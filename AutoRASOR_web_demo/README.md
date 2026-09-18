# AutoRASOR web demo

This local web app runs AutoRASOR one acquisition step at a time on synthetic phase-field micrographs. It imports the public `autorasor` package for the generator, DINOv3 feature extraction, ambiguity estimator, and v4 Matérn GP engine.

```bash
python AutoRASOR_web_demo/run_demo.py
```

The default server is available at `http://127.0.0.1:8765`. Use `--device cpu` on a machine without CUDA, `--no-browser` to suppress automatic browser launch, and `--prebuild` to generate the first field before serving. The default OPMD paths are documented in `data/README.md`; `--csv` and `--image-dir` override them.

The supported acquisition choices are:

- AutoRASOR: LFPS warmup followed by ambiguity-driven qLogNEI acquisition.
- LFPS: greedy farthest-point sampling in the low-magnification DINOv3 space.
- Random: a control for comparing stepwise metrics.

The scientific configuration uses normalized full 384-D layer-5 embeddings, cosine distance, k=10 ambiguity neighborhoods, a Matérn-5/2 ARD GP fitted by marginal likelihood, jackknife observation noise, and qLogNEI. The browser displays this resolved configuration with the run state.

`state_k` contains the archive after step `k`, the surrogate fitted on that archive, and the next recommended selection. Moving backward only scrubs cached history. Starting a new micrograph can carry prior `(x_low, y_high)` observations into the new surrogate while restricting acquisition to the new 14x14 field.

The two viewports can display the low-magnification image, high-magnification montage, DINOv3 embedding views, learned and ground-truth ambiguity, GP uncertainty, acquisition score, error, and generator parameter maps. Hover, pan, zoom, timeline navigation, and tile inspection remain synchronized across compatible layers.

## Headless mode

```bash
python AutoRASOR_web_demo/run_headless.py --output results/demo_seed42 --seed 42 --budget 40 --warmup 8
```

The headless command uses the same `Pipeline` and `DemoSession` as the browser app. It runs the campaign to its capture budget and saves configuration, observation order, pending selections, and per-step performance metrics as JSON/JSONL. It does not start a server or generate plots.

Useful options include `--device`, `--no-tta`, `--batch-size`, `--csv`, and `--image-dir`. Missing OPMD assets produce an actionable error describing the expected inputs.

## Files

```text
run_demo.py       browser entry point
run_headless.py   no-browser evaluation and result saving
pipeline.py       synthetic field generation and DINOv3 extraction
demo_core.py      stepwise session state around the public v4 engine
server.py         local JSON API and static-file server
static/           browser interface
```
