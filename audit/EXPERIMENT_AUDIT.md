# MAVRL_CBF experiment integrity and implementation audit

Date: 2026-11-04, Asia/Shanghai. Auditor: independent Codex reviewer, same-family **Type-A** review. This is neither cross-family acquittal nor GPT-5.5 certification.

Reviewed implementation: `216e9e101705cfce7ffc3ded03f1650153619622`. Upstream MAVRL: `813b8875ba00c838e08286a3f8518c6167546f74`; AvoidBench: `b9afb971ea85a98ee98125764e93513c9cf64457`. Both upstream working trees were clean when inspected. Primary files were read directly; no executor interpretation was used as evidence.

**Overall verdict: WARN. Integrity status: WARN.** The completed smoke artifacts support a functioning, learning-parameter-update pipeline, but not successful navigation or memory/speed benefits. No fabricated reference, prediction-based score inflation, or mismatched navigation result was found. One concrete resume defect and several undeclared source-fidelity differences should be addressed before further reproduction claims.

This review launched no simulator or training. It read code and logs and independently recomputed saved-data statistics and tensor/hash comparisons on CPU. `artifacts/short/` is excluded: an in-progress run is not treated as completed evidence. See `READ_ONLY_EVIDENCE.json` for recomputed checks and `REVIEWED_FILES.json` for 158 review-time file SHA-256 entries. Historical smoke checkpoints contain no `code_sha256`; the review-time commit therefore anchors this audit, not a proven historical source revision for every old artifact.

## Prioritized findings

### F1 — P1, actual defect: resume silently changes the training phase

`scripts/run.py:10` defaults `--phase` to `initial`, and `scripts/run.py:41` always passes that value. `mavrl_isaac/training.py:73`–`79` restores model, optimizer, step count and RNG but ignores the saved `phase`. Lines `81`–`82` consequently reset the environment to `empty`/`initial`; the final curriculum in `89`–`90` only runs when the caller explicitly supplies `phase='final'`. Lines `124`–`125` then save the wrong caller-supplied phase again.

Concrete trigger: `run.py train --resume artifacts/smoke/final/latest.pt --out <new-dir>` resumes final-trained weights with initial-stage tasks and rewards. `README.md:70` presents `--resume PATH` as restoration and does not state this dependency. This did **not** affect the completed smoke run: its command files contain no resume.

Action: infer phase from the loaded checkpoint when the user did not explicitly choose one; reject a conflicting explicit phase or make the change explicit. Also decide explicitly which checkpoint configuration fields are restored versus deliberately overridden. A small mocked-environment resume test can verify phase/density/reward selection without Isaac.

### F2 — P1, confirmed fidelity difference: representation data use stochastic policy actions

`mavrl_isaac/dataset.py:31` calls `actor.act(...)` without the deterministic flag; `mavrl_isaac/models.py:63`–`65` defaults to Gaussian sampling. The official collection route calls `collect_lstm_rollouts(..., deterministic=True)` at `upstream/mavrl/mav_baselines/torch/recurrent_ppo/ppo_recurrent.py:1250`; `upstream/mavrl/collect_data.py:175` selects dataset saving through that route.

The resulting VAE/LSTM training data therefore come from a different behavior policy, even if the initial weights match. This is an implemented difference, not an inference from the poor smoke success rate. It is not synthetic ground truth: the targets remain real rendered frames. `docs/SOURCE_AUDIT.md:53` describes split/reset adaptations but does not disclose the stochastic collector.

Action: use the official deterministic mean action for the reproduction collector, or explicitly label stochastic collection as an intentional variant. Preserve the historical smoke data and configuration rather than relabeling them as deterministically collected.

### F3 — P2, confirmed fidelity difference: offline LSTM learning rate is three times upstream

`configs/isaac_mavrl.json:126` sets `lstm.learning_rate=0.0003`, which is used at `mavrl_isaac/representation_training.py:77`. The official `train_lstm_without_env.py:55`–`85` does not override learning rate; the constructor default is `1e-4` at `upstream/mavrl/mav_baselines/torch/recurrent_ppo/ppo_recurrent.py:91`, applied again at line `992`. The smoke configuration also records `3e-4`.

Action: use `1e-4` for a source-matched baseline, or record the choice of `3e-4` as a deviation. This is not evidence that the learned weights are invalid; it limits claims that training hyperparameters were preserved.

### F4 — P2, evidence gap: renderer parity and historical code provenance are narrower than a general equivalence claim

`artifacts/tiled_parity.json:2`–`20` contains four exact-equality summaries, with no producer command, source-image identifiers, scene poses, or hashes. No script or test that creates this result was found in the reviewed `scripts/` and `tests/`. `scripts/benchmark_env.py:13` also writes independent/tiled snapshots to the same `depth_env_<N>_<i>.png` names. This does not prove an invalid comparison; it prevents independently reconstructing which two captures produced the four numbers.

The static camera probe establishes a center-depth reading near 4 m and a moved reading near 3 m (`logs/isaac_probe_sync.log:445`–`446`; `scripts/probe_isaac.py:16`–`32`). It does not validate the full environment RFU/FLU conversion, mounting offset, signed yaw/roll/pitch views, all 16 tiled camera identities, or scene-change reset capture. The moved reading is printed, without an assertion at line `32`.

Current `mavrl_isaac/training.py:26`–`30` saves source hashes, but the inspected historical initial/LSTM/final checkpoints lack that key; the VAE writer at `representation_training.py:42` has no source hash either. The successful artifact chain hashes prove which weights and manifest were used, not which exact historical implementation generated them.

Action: constrain existing parity claims to the observed four snapshots. For a new parity result, preserve the small comparison program and the exact source captures or their hashes, including distinctive poses and camera ordering. Record future source versions at run creation; do not retroactively attach the current commit as proven origin of the old smoke run.

### F5 — P3, confirmed task-distribution difference: the third mission orientation expands y range

For `area==2`, `mavrl_isaac/scenes.py:88`–`89` samples both start and goal y from `[0, length]`, hence `[0,40]`. The official expressions at `upstream/AvoidBench/src/avoidbench/avoidlib/src/vision_envs/avoid_vision_envs/avoid_vision_envs.cpp:202` and `207` give `start_area[0]-start_area[1] = 34` as the upper limit under `config_new_out.yaml:6`.

The three mission directions remain as documented, but their distributions are not fully source-equivalent. Action: restore `[0,34]` for this orientation, or explicitly disclose the wider range as a scene adaptation. No evidence shows this difference alone explains the smoke failures.

### F6 — P3, reporting maintenance: tracker and documentation references lag artifacts

At the reviewed snapshot, `refine-logs/EXPERIMENT_TRACKER.md:9` still says smoke is running even though all six stages and validation exist. Line `7` records the earlier eight-test log, while `logs/source_parity_tests.log` contains eleven passing tests. `docs/SOURCE_AUDIT.md:59` directs the reader to `docs/RESULTS.md`, which did not exist when inspected. These are conservative/stale bookkeeping, not invented positive results.

Action: publish the completed smoke summary with the zero-success result and update the tracker. Keep formal multi-seed and ablation rows unfinished until their required evidence exists. Independently running short training is outside this audit.

## Integrity checklist

### A. Ground-truth/target provenance — PASS

Navigation terminal events come from position, goal, elapsed time and cylinder geometry (`mavrl_isaac/environment.py:54`–`67`; `mavrl_isaac/scenes.py:26`–`30`). VAE targets are stored camera images; LSTM targets are correctly shifted stored frames (`dataset.py:27`–`28`, `representation_training.py:35`, `62`–`65`). No target is generated from a model prediction. The upstream VAE is directly imported (`models.py:9`–`12`).

The default collision geometry is consistent with the declared cylinder adaptation: cylinders span z=[0,8] and flight bounds are z=[0.4,6], so the 2D swept-circle test suffices inside the admissible volume. It is not a general finite-cylinder 3D collider. Ideal RTX depth in place of SGM, analytic cylinders in place of point clouds, and changed map generation are disclosed at `docs/SOURCE_AUDIT.md:35`–`39`, `48`–`51`. This is not an official AvoidBench benchmark score.

### B. Score normalization — PASS

Image scaling uses fixed 12 m/255 constants (`dynamics.py:43`–`46`; `Representation.step`, `models.py:35`); reconstruction MAE/MSE divide by image/pixel counts (`representation_training.py:37`–`39`, `64`–`65`). Navigation rates divide event counts by completed task count (`evaluation.py:47`–`54`). No metric denominator uses the model's own extrema.

PPO advantage standardization at `training.py:51` is the upstream training operation (`ppo_recurrent.py:657`–`660`), not a reported performance score. Arrival reward `20/TRAV` uses scene geometry, not prediction statistics; it is an explicitly approximated environment reward, not success-rate normalization. LSTM displayed MSE is a macro-average over sequences of unequal lengths (`representation_training.py:98`); it should be described that way if quoted as a dataset-level result.

### C. Result existence, numeric matching, and claims — WARN

The core smoke result files exist and their numbers match independent recomputation. The 36 saved episodes exactly reproduce all three density summaries, including Wilson intervals, mean speeds and seed lists. The evaluated policy SHA matches `final/latest.pt`; dataset policy, VAE/dataset/LSTM dependencies and all five figure source hashes also match. No phantom positive result was found.

| Density | Episodes | Success | Collision termination | Boundary termination | Timeout |
|---|---:|---:|---:|---:|---:|
| sparse | 12 | 0 | 2 | 10 | 0 |
| medium | 12 | 0 | 2 | 10 | 0 |
| dense | 12 | 0 | 3 | 9 | 0 |

Each success fraction is 0 with 95% Wilson interval [0, 0.2425007575], as stored at `artifacts/smoke/evaluation/summary.json:136`–`215`. These columns represent mutually exclusive terminal reasons, following the documented priority; they are not separate counts of every physical contact. Episode mean speeds are 0.5799293895, 0.5005606704 and 0.4913957605 m/s. No successful-episode speed exists; the stored value is correctly null.

The three benchmark rates independently recompute to 103.5984, 114.2015 and 559.7518 transitions/s from elapsed time and batch size. They are zero-action stepping benchmarks (`scripts/benchmark_env.py:14`–`18`), not full PPO throughput. The GPU string is device-wide `nvidia-smi memory.used`, not isolated per-process memory.

WARN applies to F4/F6 provenance and bookkeeping limits. The new source hash writer cannot certify earlier files retrospectively. The 8e6 value nested in evaluation `config.ppo` is the evaluation invocation's unused training default; the actual evaluated checkpoint budget is explicitly 16,384 transitions (`evaluation/summary.json:4`). Use the latter and the training directory configuration when reporting training budget.

### D. Dead metrics and actual execution — PASS

The evaluation entry at `scripts/run.py:45`–`47` calls `evaluate`; `wilson` is called at `evaluation.py:49`; navigation outcomes and sampled speed/depth/actions are written at lines `56`–`60`. VAE loss, temporal reconstruction, and memory-reset diagnostics are reached by the training entry points (`scripts/run.py:29`–`32`; `representation_training.py:35`, `95`, `110`–`117`). Their outputs exist.

Initial PPO logs contain 16 updates/8,192 transitions and 80 boundary-terminated episodes; final PPO contains 8 updates/16,384 transitions and 313 episodes (147 collision, 166 boundary). All logged numeric values are finite and per-update episode counts/success fractions match episode JSONL. Both VAE and LSTM contain epoch 0 evaluation plus 10 training epochs. Stored initial policy weights change between 5,120 and 8,192 steps while representation tensors do not; LSTM recurrent/projection/decoder tensors change from initial, its encoder exactly matches VAE best, and final representation exactly matches LSTM best. These observations rule out merely saving untouched random weights for those components.

`scripts/validate_artifacts.py:11`–`17` tests two reloads of the same checkpoint; its name `checkpoint_load_action_identity` is justified, but it is not a pre-save/post-save trajectory test. The reviewer inspected existing tests/logs, did not rerun them, and did not numerically rerun full neural training or simulation. Missing producer evidence for tiled parity is reported in C/F4, not assumed to be executed code.

### E. Scope versus evidence — WARN for evidence sufficiency; reporting is appropriately qualified

One training seed, 12 tasks per density, and short budgets do not establish C1/C2 in `refine-logs/EXPERIMENT_PLAN.md:9`–`10`. There are no successful smoke navigation episodes. The formal three-seed and paired retraining ablations remain planned. A hidden-state reset reconstruction diagnostic cannot establish a separately trained no-memory navigation advantage; its own artifact explicitly says so (`artifacts/smoke/lstm/memory_diagnostic.json`, final interpretation field).

The current text correctly calls results smoke/short training and disclaims convergence/original AvoidBench reproduction (`evaluation.py:59`; `evaluation/summary.json:218`; `README.md:3`, `35`, `91`). No inflated “comprehensive/robust” completed-result claim was found. This WARN is an evidence ceiling, not a finding of dishonest wording.

The dataset contains 78 contiguous files and 4,029 retained frames (3,139 train; 890 validation), from 36 train and 11 validation scene seeds. All NPZ scene IDs, frame counts, dtypes/shapes and state finiteness agree with the manifest. There is no scene-seed intersection or identical full sequence across splits. Evaluation seeds are disjoint from dataset seeds and from all completed initial/final training episode seeds. The environment generates training seeds from the unrestricted integer range (`environment.py:19`), so formal future experiments should explicitly verify disjointness rather than assume the fixed evaluation offset mathematically reserves seeds. No observed leakage was found in this snapshot.

### F. Evaluation type — PASS: simulation_only; reconstruction is self_supervised_proxy

Navigation and rendering/throughput tests are **simulation_only**. VAE/LSTM reconstruction and reset diagnostics are **self_supervised_proxy on simulated observations**, because observed frames supervise representation learning without semantic navigation ground truth. They are not `synthetic_proxy` generated from model outputs, not a real-world dataset benchmark, and not human evaluation. Geometric success checks provide simulator truth, not real-world validation.

## Numerical equivalence: what is and is not tested

| Component | Evidence and justified conclusion |
|---|---|
| VAE/encoder/decoder shape and forward computation | Direct upstream VAE import; `tests/test_source_parity.py:13`–`24` loads equal weights into official/local modules and compares outputs exactly. Existing eleven-test log passes. |
| LSTM recurrence and episode reset mask | `tests/test_source_parity.py:25`–`37` extracts the actual upstream `_process_sequence` AST and compares output/states with tolerances. This is direct source-level numerical evidence. |
| GAE terminal isolation | Hand-constructed numerical case at `tests/test_source_parity.py:38`–`41`; validates that example, not general upstream buffer equivalence. |
| Dynamics, reward, collision, terminal overwrite | Formula/source comparison supports the implementation; `tests/test_core.py:14`–`44` checks selected analytic cases. No compiled C++ versus NumPy batch equivalence test exists. In particular, combined nonzero roll/pitch/yaw trajectories are not established by these tests. |
| PPO surrogate, cached features, action distribution | Static comparison supports cached frozen latent, tanh mean, unclipped stored samples, GAE and PPO formula fidelity (`training.py:44`–`62`, `92`–`110`; upstream `ppo_recurrent.py:415`–`481`, `649`–`721`). There is no identical-batch optimizer-update comparison. F2/F3 remain separate behavioral/hyperparameter differences. |
| Full camera/frame conversion and tiled capture | Single-camera axial probe and four parity rows are limited observations. Signed yaw/roll/pitch, offset, ordering and reset synchronization still need targeted simulator comparisons before broad equivalence claims. No camera bug is asserted from missing tests alone. |
| Dataset episode reset | Buffers are appended before stepping and flushed on `done`; retained `group` labels change only after flush (`dataset.py:27`–`36`). Despite the inaccurate “Flush BEFORE environment autoreset” comment, the old scene-seed group makes the labels episode-local. No cross-episode target defect was found. |

Minor documentation correction: `docs/SOURCE_AUDIT.md:33` says “left-multiply heading”; the implemented and official product is `q_pitch_roll * q_heading` (`dynamics.py:19`; official environment line `377`). The code agrees with the source; the multiplication-order prose should be made unambiguous.

## Claim impact and next actions

| Claim | Assessment |
|---|---|
| Six-stage smoke pipeline produced real artifacts and parameter updates | Supported, with historical source-version provenance limitation. |
| Unseen-cylinder navigation is successfully reproduced (C1) | Unsupported by the completed smoke run: 0/36 success; no conclusion about sufficient-budget training. |
| Past reconstruction or adaptive speed improves navigation (C2) | Unsupported: no paired equal-budget retraining/navigation ablation. |
| Fixed-source MAVRL equations and network structure are substantially ported | Supported with qualifiers: F2/F3/F5 differences and limited equivalence tests. |
| Original AvoidBench benchmark performance or real-world capability reproduced | Unsupported and appropriately disclaimed. |

Fix F1, resolve/disclose F2/F3, and update the source adaptation record for F5. Then perform only the targeted checks needed for those changes and the camera claims. Keep historical artifacts immutable and label the current run as smoke. Formal success, memory, and speed conclusions require the already-planned experiments, not additional speculative infrastructure.
