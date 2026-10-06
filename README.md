# Predicted NFL Defenders That Play Quarters Like Real Ones

Code for **c4sim**, the model in the abstract of the same title. Given the snap alignment, the coverage call, the routes the receivers
actually ran and the quarterback's movement, c4sim predicts every coverage defender's path from the snap to the pass (up to 80 frames,
8 s at 10 Hz). It is scored on 2018 Cover 4 (Quarters) plays.

## Results

Pooled over four folds: 1,866 Cover-4 passing plays in 248 games, each predicted by a model that never trained on its fold's 62 games.
Each cell lists the real defense, then the standard, midpoint and route-break outputs. Error is per-coordinate RMSE in yards.

| Measure (real, standard, midpoint, route-break) | Pooled |
|---|---|
| Mean error (yd) | —, 1.09, 1.10, 1.13 |
| Median play (yd) | —, 0.72, 0.72, 0.73 |
| Openness at throw (yd) | 4.23, 4.25, 3.97, 3.70 |
| Tightest receiver (yd) | 1.77, 1.91, 1.80, 1.67 |
| 1 s after a cut (yd); 164 breaks | 2.55, 3.57, 3.20, 2.84 |
| #2 read: safety 5+ yd deeper, #2 vertical / flat (%) | 55/25, 49/13, 44/12, 39/12 |
| Solo: backside safety 2+ yd toward trips, #3 vertical / short (%) | 33/14, 19/10, 22/10, 24/10 |

A coverage classifier that sees only player movement (fold 0) calls Quarters on 92.8% of the real defense's plays, on 94.3 / 94.1 /
93.2% of the standard / midpoint / route-break outputs', and on 69.5% for a defense frozen at the snap.

## How the model works

**Inputs.** The model sees up to 8 defenders, 5 route runners and the QB per play, in yards with the offense moving toward +x.
- **Defender features** per frame (51): his five nearest route-set players (offset, distance, velocity, closing speed), his snap
  spot, the QB's offset and drop, his depth off the line, play-level counts, the coverage call and play action.
- **Route runners** (depth, lateral offset, velocity per frame).
- **Coded Cover-4 rules** (`src/prep/rule_inputs.py`): 12 options per defender (5 receiver corridors, 7 zone roles). Each is a
  68-number token and an 11-step preview from physics solves of how he would reach the option from his snap state. Alongside them:
  zone-landmark reach times and his pre-snap boundary state.
- **Analog plays**: the matched defenders' mean path in the 64 nearest training plays of the same coverage call.
- **Orientation, roster position and blocker flags.**

**Encoders.**
- **Defender-frame token:** the inputs projected to width 192.
- **Rule-option attention:** each defender's boundary state and first frame query his option tokens (with a GRU summary of each
  preview, plus a learned "keep doing what you do" token), giving a 64-d reaction summary that is broadcast to every frame.
- **Route encoder:** a 2-layer transformer over each runner's whole path.

**Scene transformer.** The defender, runner and QB tokens at every frame pass through blocks that attend over time for each player,
then over players at each frame, with a learned bias and messages from pairwise geometry. Three refinement rounds of two blocks each
re-measure the geometry from the previous round's proposal.

**Heads.** A readout gives each defender's own displacement. A receiver-anchored head mixes in a weighted pull toward each runner (plus
a learned cushion), and a confidence head gives a log-variance. The third round is the **standard output**.

**Route-break output.** A detached alternative final round (two blocks and its own heads), trained with an extra openness penalty
whenever it leaves a closely covered runner more open than the real defense did. Because it is detached, it never changes the
standard output. The **midpoint** averages the two.

**Losses.**
- Gaussian negative log-likelihood plus a late-weighted squared error (each track's last frame counts 2× the first).
- Squared error on rounds 1–2.
- A reaction loss on the first second of real movement.
- A training-only rule-teacher loss.
- The openness penalty (route-break output only).

The weights are in `src/recipe.py`.

**Training.**
- **Data:** every zone-coverage dropback (2018, 2021–2023) outside the fold's 62 games, about 29,700 plays.
- **Optimization:** a saved random initialization, seed 42; AdamW at 3e-4 and 16 plays per update; a 36-epoch cosine schedule
  stopped after 28.
- **Evaluated weights:** an EMA of the weights.
- **Augmentation:** mirroring and entity dropout.
- **Size:** 4.5 M parameters.

**Inference.** Each play is predicted as recorded and mirrored, and the two are averaged; each track is then smoothed.

## Running it

**Setup.**
- Python 3.10 and `pip install -r requirements.txt`.
- Unpack the data below under `$DATA_ROOT/raw/`.
- Run from the repo root with `DATA_ROOT` and `OUT_ROOT` set.

| Data | Source | Folder under `raw/` |
|---|---|---|
| 2018 | [Big Data Bowl 2021](https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data) | `nfl-big-data-bowl-2021/` |
| 2018 coverage calls | [nflverse `pbp_participation_2018.csv`](https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation) | `nflverse/` |
| 2021 | [Big Data Bowl 2023](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data) | `nfl-big-data-bowl-2023/` |
| 2022 | [Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) (files now in [this archive](https://www.kaggle.com/datasets/alexandermeau/nfl-big-data-bowl-archived-data-2025)) | `nfl-big-data-bowl-2025/` |
| 2023 | [Big Data Bowl 2026 (analytics)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data) | `nfl-big-data-bowl-2026-analytics/` |

| Stage | Command |
|---|---|
| Frames, plays, tracks, windows, defenders | `for s in frames plays tracks windows defenders; do python -m src.prep.$s; done` |
| Coded-rule inputs (physics solves) | `python -m src.prep.rule_inputs` |
| Records, per-fold inputs | `python -m src.prep.records`, `python -m src.prep.folds` |
| Train fold K | `scripts/train_fold.sh K` |
| Predict, baselines | `python -m src.predict --fold K`, `python -m src.eval.baselines --fold K` |
| Table | `python scripts/reproduce_table.py` |

**Training** needs `initial_state.pt` from the release in `$DATA_ROOT/prepared/`. The evaluation-only models (coverage classifier,
completion probability) are in the release too; the scorers' other evaluation inputs aren't built by this code yet.

## Notes

- **Folds:** split by game (`metadata/folds.csv`). **Population:** `metadata/plays.csv`.
- **Scoring mask:** frames after a scramble breakdown aren't scored on 54 plays (`metadata/scoring_mask.csv`).
- **Coded rules, analog plays and the coverage call are inputs**, at prediction time too: the model answers how a given coverage
  responds to a given play. The rule teacher is a training-only loss.
- **Epoch selection:** each output is reported at its best epoch on the held-out fold.
- **Upstream models:** LightGBM models pick the coverage defenders and supply 2018 play action.

## License

Code: MIT. The Big Data Bowl data are subject to each Kaggle competition's rules and aren't included; the 2018 coverage calls are
nflverse data.
