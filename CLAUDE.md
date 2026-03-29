# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All ML commands run from inside `LodeRunnerMachineLearning/`.

**Train the model:**
```bash
cd LodeRunnerMachineLearning
python main.py -c configs/lstm_config.json
```

**Generate a level** (requires a trained `.hdf5` checkpoint in `generate/selectedModels/`):
```bash
cd LodeRunnerMachineLearning
python generate/main.py
```

**Run tests:**
```bash
cd LodeRunnerMachineLearning
python -m unittest tests/generate_tests.py
```

**Start TensorBoard** (after training, from repo root):
```bash
tensorboard --logdir=LodeRunnerMachineLearning/experiments
```

Training outputs checkpoints and TensorBoard logs under `LodeRunnerMachineLearning/experiments/<timestamp>/LodeRunner_LSTM/`.

## Architecture

### LodeRunnerMachineLearning (Python/Keras)

The ML pipeline follows a base-class/subclass pattern across four components:

- **`data_loader/`** — Loads 150 level `.txt` files from `utils/Others/Levels_DataSet/`. Vectorizes each 16×28 tile grid into one-hot vectors (10 tile types). Traversal is **column-major snake order** (top→bottom on even columns, bottom→top on odd columns), not row-major.
- **`models/`** — `LodeRunnerModel` builds a `CuDNNLSTM(64) → Dense(10) → Softmax` sequential model. The input shape is `(1, 10)` — one tile at a time.
- **`trainers/`** — Wraps Keras `model.fit()` with `ModelCheckpoint` and `TensorBoard` callbacks. Hard-coded reshape: `(44700, 1, 10)` = 100 levels × 448 tiles × 1 step.
- **`tester/`** — Wraps `model.evaluate()`. Hard-coded reshape: `(22350, 1, 10)` = 50 levels × 447 tiles.

Config is loaded from a JSON file via `python-bunch` into a `Bunch` object (dot-accessible dict). `process_config()` appends timestamped `tensorboard_log_dir` and `checkpoint_dir` paths.

### generate/main.py

Standalone inference script. Loads a saved `.hdf5` model, generates a 16×28 map token-by-token using weighted random sampling from softmax output, then applies a series of post-processing repair functions to enforce playability constraints (solid bottom floor, ladders touching ground, handrails extended, pickup/enemy counts capped, player placed on row 14).

### Tile token legend

| Char | Meaning |
|------|---------|
| `#`  | Solid block |
| `@`  | Brick (diggable) |
| `H`  | Ladder |
| `-`  | Handbar (rope) |
| `X`  | Trap |
| `S`  | Solid (variant) |
| `$`  | Pickup (gold) |
| `0`  | Enemy |
| `&`  | Player |
| ` `  | Empty |

### LodeRunnerApplication (JavaScript)

A browser-based Lode Runner game built with CreateJS. Entry point is the HTML file served from the app root; game logic is split across `lodeRunner.*.js` modules (runner, guard, edit, menu, etc.). Level data for 5 game variants is embedded in `lodeRunner.wData.js` and `lodeRunner.v.*.js` files. No build step — runs directly in a browser or via IIS (see `Web.config`).
