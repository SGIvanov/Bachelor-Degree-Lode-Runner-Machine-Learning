# Procedural Level Generation in Games Using Recurrent Neural Networks

Bachelor's Degree Thesis

---

## Overview

This project investigates **Procedural Content Generation (PCG)** using machine learning, specifically applying **Long Short-Term Memory (LSTM)** networks to generate new playable levels for the classic arcade game *Lode Runner*.

Traditional PCG relies on hand-crafted rules, grammars, or search algorithms to generate content. This work takes a data-driven approach: instead of encoding domain knowledge explicitly, an LSTM is trained on existing levels and learns structural patterns implicitly. The generated output is then passed through a lightweight repair pipeline to enforce playability constraints that the neural network alone cannot reliably guarantee.

The two core goals of the generation are **diversity** — levels that differ meaningfully from the training set — and **playability** — levels that are completable by a player. The main challenge acknowledged by the thesis is the small dataset (150 levels), which limits diversity and increases the risk of the model overfitting to seen patterns.

---

## Repository Structure

```
├── LodeRunnerMachineLearning/   # Python/Keras ML pipeline
│   ├── configs/                 # Training configuration (JSON)
│   ├── data_loader/             # Level parsing and vectorization
│   ├── models/                  # LSTM model definition
│   ├── trainers/                # Training loop with callbacks
│   ├── tester/                  # Evaluation on held-out levels
│   ├── generate/                # Inference and post-processing repair
│   │   └── selectedModels/      # Saved .hdf5 checkpoint used for generation
│   └── utils/
│       └── Others/
│           ├── Levels_DataSet/  # 150 extracted Classic Lode Runner levels
│           └── Commands/        # CLI usage reference
└── LodeRunnerApplication/       # Browser-based Lode Runner game (JavaScript/CreateJS)
```

---

## Technical Decisions

### 1. Game Choice — Lode Runner

*Lode Runner* was chosen because its levels are fully representable as a 2D character grid of fixed size (28 columns × 16 rows), making them straightforward to encode as sequences. The game has a small, well-defined tile vocabulary (10 types), which keeps the output space manageable for a classification model.

### 2. Level Representation

Each tile is encoded as a **one-hot vector of length 10**:

| Character | Tile Type        |
|-----------|------------------|
| `#`       | Solid block      |
| `@`       | Brick (diggable) |
| `H`       | Ladder           |
| `-`       | Handbar (rope)   |
| `X`       | Trap             |
| `S`       | Solid variant    |
| `$`       | Pickup (gold)    |
| `0`       | Enemy            |
| `&`       | Player           |
| ` `       | Empty space      |

Representing tiles as one-hot vectors rather than integer indices treats the problem as **multi-class sequence classification** and avoids implying any ordinal relationship between tile types.

### 3. Traversal Order — Column-Major Snake

Levels are read in **column-major snake order**: top-to-bottom on even columns, bottom-to-top on odd columns. This means the LSTM sees tiles that are spatially adjacent vertically before moving to the next column, rather than reading left-to-right row by row.

The rationale is that Lode Runner's structural patterns (platforms, ladders, ropes) are more coherent as vertical structures than as horizontal rows. A snake traversal also means every consecutive pair of tiles in the sequence is a direct neighbour in the grid, giving the LSTM a better spatial locality signal.

### 4. Model Architecture

```
CuDNNLSTM(64) → Dense(10) → Softmax
```

The model is intentionally shallow — a single LSTM layer with 64 units feeding directly into a 10-class softmax output. Given the small dataset, a deeper model would overfit more severely. The `CuDNNLSTM` variant was used for GPU-accelerated training on NVIDIA hardware.

The task is framed as **next-tile prediction**: given the one-hot vector of the current tile, predict the probability distribution over the 10 tile types for the next position in the snake sequence. This is analogous to character-level language modelling applied to level grids.

**Training configuration:**
- Dataset: 100 levels for training, 50 for testing
- Epochs: 50
- Batch size: 16
- Optimiser: Adam (lr = 0.001)
- Loss: Categorical cross-entropy
- Validation split: 25% of training data
- Best checkpoint saved by minimum `val_loss`

### 5. Generation — Weighted Sampling

During inference, the model generates one tile at a time in the same snake order used during training. At each step, the softmax output is treated as a **probability distribution** and the next tile is **sampled proportionally** (not argmax-selected). This introduces stochasticity, producing different levels on each run and avoiding the collapse to a single most-probable sequence.

The generation is seeded with a random one-hot starting tile, and the model autoregressively feeds its own predictions as inputs for subsequent steps.

### 6. Post-Processing Repair Pipeline

Because the LSTM has no explicit notion of game rules, a deterministic repair pass is applied after generation to produce a structurally valid level. The repairs are applied in this order:

1. **Solid bottom floor** — row 15 is forced to all `#` blocks, ensuring the player has ground to stand on.
2. **Clear second-to-last row** — row 14 is cleared of solid tiles to leave a usable floor area.
3. **Clear top row** — row 0 is cleared to avoid a ceiling.
4. **Fill gaps** — single-tile-wide gaps enclosed by solid blocks on both sides and below are filled with `#`, preventing unreachable isolated voids.
5. **Ladders touch ground** — each ladder segment is extended downward until it reaches a solid tile, ensuring it is accessible.
6. **Ladder structural cleanup** — adjacent ladder tiles get surrounding tiles adjusted to keep paths unobstructed.
7. **Handbars extended** — rope segments are extended one tile in each direction if the adjacent cell is empty, increasing reachable area.
8. **Pickups touch ground** — gold tokens (`$`) are moved down to rest on the nearest solid surface below them.
9. **Pickup count capped** — at most 4 gold tokens are kept; excess are removed.
10. **Enemy count enforced** — exactly 2 enemies (`0`) are placed; extras removed, shortfall filled from empty cells.
11. **Player placed** — all existing player tokens (`&`) are removed and one is placed in the first empty cell of row 14 (the usable floor).

This hybrid approach — neural generation followed by rule-based repair — is the core architectural decision of the thesis. The network handles global structure and variety; the rules enforce local constraints that require hard guarantees.

---

## Results

Training metrics (loss and accuracy) are logged to TensorBoard under `experiments/`. The best model checkpoint achieved a validation loss of **~1.27** at epoch 1 and is stored at `generate/selectedModels/LodeRunner_LSTM-01-1.27.hdf5`.

Screenshots of training curves and example generated maps are archived in `Iteratia 1 , 1 Timestamp, 0 repair/`.

---

## Running the Project

**Train:**
```bash
cd LodeRunnerMachineLearning
python main.py -c configs/lstm_config.json
```

**Generate a level:**
```bash
cd LodeRunnerMachineLearning
python generate/main.py
```

**Run tests:**
```bash
cd LodeRunnerMachineLearning
python -m unittest tests/generate_tests.py
```

**TensorBoard:**
```bash
tensorboard --logdir=LodeRunnerMachineLearning/experiments
```

**Dependencies:** Python 3, Keras (TensorFlow backend), NumPy, `python-bunch`.

---

## Web Game

`LodeRunnerApplication/` contains a fully playable browser-based implementation of *Lode Runner* built with JavaScript and [CreateJS](http://www.createjs.com). It includes 5 game variants (Classic, Professional, Revenge, Fan Book, Championship), a level editor, and two visual themes (Apple-II and Commodore 64). No build step is required — open the HTML file directly in a browser or serve it via IIS.
