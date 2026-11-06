# Actor/critic LSTM serialization and reload review

Date: 2026-11-04. Reviewer: independent Codex, same-family **Type-A** experiment audit; not cross-family certification. No GPU, simulator, or training was started. All checkpoint reads used `map_location='cpu'` with CUDA hidden; `torch.cuda.is_initialized()` remained false.

**Finding: the official initial PPO shares the actor LSTM, but the supplied save/reload → offline LSTM → retrained PPO path creates and retains a separate critic LSTM. The local single-LSTM implementation does not reproduce that latter path.** This is a **P1 source-fidelity defect**, not evidence of fabricated results. The original review's broad treatment of shared actor/critic memory was incomplete; this report supersedes that aspect of `EXPERIMENT_AUDIT.md` and the unqualified statement at `docs/SOURCE_AUDIT.md:19`.

The local implementation reviewed here is the version at `6c2e89ca8ce062fb2a583b3f2d3ff13607664f25`. The executor began changing working-tree code after the finding was delivered. Those concurrent repairs are **not reviewed by this report**. To keep the numerical evidence independent of those edits, the final CPU comparison executed the local model definitions directly from `git show 6c2e89c:mavrl_isaac/models.py`. Source hashes, checkpoint metadata and numerical results are preserved in `CRITIC_SOURCE_EVIDENCE.json`; hashes captured from the changing working tree are separately labeled as unreviewed.

## Exact construction and serialization path

| Stage | Actor/critic recurrent relationship | Primary evidence |
|---|---|---|
| Initial PPO, `retrain=False` | One actor LSTM; critic consumes its detached output. | `upstream/mavrl/mav_baselines/torch/recurrent_ppo/ppo_recurrent.py:257`–`272` explicitly passes `shared_lstm=True`, `enable_critic_lstm=False`. |
| Policy construction defaults | Independent actor and critic LSTMs unless overridden. | `upstream/mavrl/mav_baselines/torch/recurrent_ppo/recurrent/policies.py:817`–`818` defaults to `False`/`True`; lines `848`–`849` forward those values. Base recurrent class has the same defaults at `100`–`101`; critic LSTM is created at `170`–`176`. |
| Policy serialization | Recurrent sharing flags are not explicitly serialized by the MAVRL subclass. | Neither recurrent class overrides `_get_constructor_parameters` or `save`; superclass import is `policies.py:9`, inheritance at `36`. Official PPO saves with `self.policy.save` at `ppo_recurrent.py:872`; offline LSTM does so at `1037`. |
| Offline LSTM reconstruction from initial checkpoint | Rebuilding from metadata missing those flags creates a new separate critic LSTM. Initial shared checkpoint has no `lstm_critic.*` tensors, and `strict=False` tolerates their absence. | `upstream/mavrl/train_lstm_without_env.py:43`–`52` supplies `features_dim`, reconstruction settings and saved `data`, without explicit sharing flags. |
| Offline optimization | Reconstruction trains actor LSTM, projection and decoder; the separate critic LSTM is unused by the reconstruction loss. | `policies.py:614` uses only `self.lstm_actor`; lines `617`–`647` perform reconstruction. `ppo_recurrent.py:1013`–`1020` backpropagates that loss. |
| Final PPO reconstruction | Reload again defaults to independent LSTMs, then loads both sets if the LSTM checkpoint contains them. | `upstream/mavrl/train_policy.py:128`–`149` reconstructs from `data`; `--nocontrol` removes MLP/output weights but does **not** remove `lstm_critic.*`. |
| Final PPO setup, `retrain=True` | Keeps the already-built policy; the initial explicit sharing flags do not run. | `ppo_recurrent.py:185`–`186` assigns the supplied policy; the policy-construction branch at `257` is guarded by `if not self.retrain`. |

The serializer's behavior is corroborated by actual artifacts, not inferred only from a modern dependency. Both distributed checkpoints contain the same 16 metadata keys:

```text
observation_space, action_space, normalize_images, net_arch, activation_fn,
use_sde, log_std_init, squash_output, full_std, use_expln, lr_schedule,
ortho_init, optimizer_class, optimizer_kwargs,
features_extractor_class, features_extractor_kwargs
```

Neither contains `shared_lstm` or `enable_critic_lstm`; neither stores `lstm_hidden_size`, `n_lstm_layers` or a custom recurrent reconstruction specification. For comparison, the installed SB3 2.9.0 `BaseModel.save` at `/home/lz/miniconda3/envs/lz_env/lib/python3.11/site-packages/stable_baselines3/common/policies.py:158` writes `state_dict` plus `_get_constructor_parameters()`, and the inherited `ActorCriticPolicy` method at line `537` contains generic feedforward policy arguments rather than recurrent flags. **The installed version is not asserted to be the original training environment**; the checkpoint metadata itself establishes the missing flags for the published weights.

The absence of an original initial-stage or offline-LSTM-stage checkpoint in this review limits historical reconstruction: the moment at which the critic was first initialized follows the shown code path. The existence of separate actor/critic recurrent weights in the two published final policies is directly verified.

## Actual public checkpoint evidence

| Checkpoint | SHA-256 | Recurrent parameters |
|---|---|---|
| `upstream/mavrl/ckpts/available_policy/iter_00540.pth` | `a530396e060fb31f431344d701fd27241e1afed59d8b307a6f5b0a619a1a7e80` | Four `lstm_actor.*` and four `lstm_critic.*` tensors. |
| `upstream/mavrl/ckpts/available_policy/iter_00640.pth` | `799ec5e313d4c00f48292b9023746ee3e76e6ef7c200442ea74007b0c27ada57` | Four `lstm_actor.*` and four `lstm_critic.*` tensors. |

Each LSTM has `weight_ih_l0=[1024,64]`, `weight_hh_l0=[1024,256]`, and two `[1024]` biases. Actor and critic tensors differ, with respective maximum absolute differences 0.2022562325, 0.2721535563, 0.3762779236 and 0.3026920557. They are not aliases or duplicated copies of a shared LSTM.

Between checkpoint 540 and 640, all four actor LSTM tensors and all four critic LSTM tensors are exactly equal, whereas the policy and value MLPs each contain changed tensors. This is consistent with the actual PPO gradient path: both recurrent features are computed under `no_grad` at `ppo_recurrent.py:415`–`420`, stored separately at `469`–`470`, and supplied separately to `evaluate_actions` at `649`–`652`. That function applies the respective MLPs (`policies.py:482`–`489`) without recomputing either LSTM. Thus PPO does not train the separate critic LSTM either.

Under the default initial-checkpoint → offline-LSTM path, a newly initialized critic LSTM receives no reconstruction gradients and remains frozen during subsequent cached-feature PPO. That describes the code's behavior even if it originated as a serialization omission. It should not be silently replaced with shared actor memory while claiming exact source behavior. The two published final checkpoints alone do not prove the exact seed or initialization history of their critic tensors.

## Why the local final PPO differs

At the reviewed local version, `mavrl_isaac/models.py:24` creates only `Representation.lstm`. `Representation.step` at lines `34`–`41` returns one latent and one hidden/cell pair. `ActorCritic.distribution_value` at line `61` passes the same input to both MLPs.

`mavrl_isaac/representation_training.py:72` reloads that same single-LSTM representation; reconstruction at `52`–`65` never introduces an independent critic. `mavrl_isaac/training.py:70`–`71` reloads it for final PPO. Rollouts at `95`–`99`, bootstrap at `105`–`106`, and optimization at `49` consequently all use actor-memory features for value estimation.

This matches the official **initial PPO sharing choice**, but not the official reloaded policy used for final PPO. The difference changes value predictions, GAE advantages/returns and therefore actor updates. It is not merely an unused extra module or a naming difference. Explicitly freezing the single LSTM does not remedy the missing independent critic state/features.

The old smoke initial, representation and final checkpoints all contain only the four `lstm.*` tensors; there is no saved critic recurrence to recover. Corrected future training should preserve those older runs as the shared-critic variant rather than claim their final phase used the newly established source path.

## Public-weight importer: actor conversion versus full-policy conversion

`scripts/import_official_checkpoint.py:24` maps only `lstm_actor.*` into the single local LSTM. Lines `29`–`33` map the policy/value MLPs and output heads, but there is no mapping for the four `lstm_critic.*` tensors. `strict=True` applies to each **selected destination submodule**; it does not check coverage of all source checkpoint keys. The line `4` claim “all weights load strictly” is therefore incomplete as a statement about the full official policy.

The practical consequences differ:

- **Deterministic navigation actions:** the actor uses encoder → actor LSTM → actor MLP → action head, so dropping a separate critic does not by itself change actions. The official inference function likewise uses only actor recurrence (`policies.py:415`–`419`). Local evaluation discards returned values (`mavrl_isaac/evaluation.py:28`–`29`). This finding alone therefore does not invalidate actor-only zero-shot navigation outcomes.
- **Critic values or further PPO training:** the converted value MLP is fed the wrong recurrent feature. It is not a faithful full actor-critic conversion, and cannot support value parity or source-equivalent resumed optimization claims.

A CPU-only differential fixture verifies this distinction. The first two images/states in `artifacts/smoke/dataset/train/sequence_00000.npz` were treated as two independent zero-state episodes. Source-shaped actor/critic modules were loaded directly from the raw official 540 tensors. The comparison used local model definitions from the frozen git revision and the existing converted checkpoint, without rendering or optimization.

| Quantity | Result |
|---|---:|
| Maximum actor mean difference, source versus converted | **0.0** |
| Source separate-critic values | `[0.0468327515, -0.0938983634]` |
| Local shared-latent values | `[0.0873402432, -0.0638476685]` |
| Maximum value difference | **0.0405074917** |
| Maximum actor/critic recurrent latent difference | 0.2524030805 |

This small fixture establishes a concrete value mismatch and actor equality for those inputs; it is not a simulator performance evaluation or a full old-runtime integration test. Exact input/checkpoint hashes and the complete metadata are in `CRITIC_SOURCE_EVIDENCE.json`.

## Required correction or explicit scope choice

For a source-faithful staged baseline, retain shared recurrence for initial PPO and introduce/preserve the independently initialized critic recurrence when reconstructing the policy for offline LSTM training. Train reconstruction through the actor recurrence only; preserve the critic weights in the representation checkpoint. Final PPO must maintain separate actor/critic hidden states and cached features, with both recurrent modules frozen. The importer should load the published critic tensors if it claims full-policy conversion; cloning the actor weights into the critic would not reproduce the public checkpoints.

Alternatively, intentionally keep shared recurrence across all stages and explicitly describe it as a correction/variant of the published serialization path. That is a valid experimental choice, but it requires separate reporting and cannot be called the unchanged official final-PPO algorithm. An actor-only transfer importer may omit the critic if it is explicitly labeled and its value output is not used as official-policy evidence.

The focused verification is a CPU save/reload test preserving stage topology, separate reset masks/states, source-weight actor/value comparisons, and identical-batch PPO comparisons using distinct cached actor/critic inputs. No new GPU training is needed to establish the structural correction. Whether the correction improves navigation needs future experiment evidence; no such improvement is asserted here.
