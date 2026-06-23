# PPO-based Dispatch with GNN ETA

This repository contains a synthetic last-mile dispatch simulator and a set of learning pipelines for courier-order assignment. The current codebase combines:

- time-dependent travel-time simulation on a synthetic road grid
- ETA estimation with graph neural networks
- PPO-based dispatch policies
- masked multi-discrete action selection for feasible rider-order-route insertion
- joint policy training with ETA-aware supervision and uncertainty-aware risk terms

The project is structured as a research codebase rather than a polished package. The README below focuses on the workflows that match the code as it exists today.

## What Is In The Repo

- `sim/`: core dispatch environments
  - `env_dispatch.py`: base simulator
  - `env_dispatch_ppo.py`: Gymnasium PPO environment with structured observations and action masks
- `models/`: neural models
  - `eta_gnn.py`: edge travel-time GNN
  - `dispatch_bipartite_gnn.py`: rider-order bipartite scoring model
  - `joint_encoder.py`: joint node-set encoder used by PPO policies
- `train/`: training and evaluation entry points
- `data_gen/`: synthetic CSV generation utilities
- `configs/`: YAML experiment settings used by the newer evaluation workflow
- `scripts/`: analysis, profiling, and plotting helpers
- `outputs/`: example CSV assets, pretrained checkpoints, and exported results

## Environment Setup

Python `3.9` is the safest target for this repository.

```bash
git clone https://github.com/andreapan1999-cloud/PPO-based-dispatch-with-GNN-ETA.git
cd PPO-based-dispatch-with-GNN-ETA

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` already includes `stable-baselines3`, `sb3-contrib`, `torch-geometric`, `gymnasium`, and plotting dependencies.

## Recommended Workflows

There are two practical ways to use the repository.

### 1. Quick Start With The Bundled `outputs/` Assets

Most training scripts expect CSV files such as `outputs/orders.csv`, `outputs/riders.csv`, `outputs/nodes.csv`, and `outputs/travel_times.csv` to already exist. This repository already includes sample assets, so the fastest path is to use them directly.

Smoke-test baseline PPO:

```bash
python -m train.train_dispatch_ppo --seed 0 --timesteps 2048
```

Train masked PPO with the joint encoder:

```bash
python -m train.train_dispatch_ppo \
  --maskable \
  --seed 0 \
  --timesteps 50000 \
  --target_kl 0.05 \
  --tb
```

Train the newer joint masked PPO variant:

```bash
python -m train.train_dispatch_maskable_ppo \
  --seed 0 \
  --timesteps 50000 \
  --target_kl 0.05 \
  --tb \
  --eta_uncertainty
```

Notes:

- `train/train_dispatch_ppo.py` saves checkpoints like `outputs/ppo_seed0.zip` or `outputs/maskable_seed0.zip`.
- `train/train_dispatch_maskable_ppo.py` saves a run directory under `outputs/runs/` and also exports a convenience checkpoint under `outputs/`.
- TensorBoard logs are written only when `--tb` is passed.

### 2. Evaluation And Experiment Sweeps With `configs/*.yaml`

The newer evaluation flow takes a YAML config and automatically generates a dataset specific to that config before running policy evaluation.

Example: evaluate a pretrained checkpoint on the main final setting.

```bash
python -m train.eval_dispatch_ppo \
  --config configs/final_main.yaml \
  --model outputs/maskable_joint_ueta_seed2.zip \
  --vecnorm outputs/vecnormalize.pkl \
  --episodes 50 \
  --deterministic \
  --tag final_main
```

Useful evaluation modes:

- greedy baseline:

```bash
python -m train.eval_dispatch_ppo \
  --config configs/final_main.yaml \
  --greedy_only \
  --episodes 50 \
  --tag greedy_final_main
```

- random baseline:

```bash
python -m train.eval_dispatch_random \
  --episodes 50 \
  --seed_base 2000 \
  --vecnorm outputs/vecnormalize.pkl
```

- action-mask sanity check:

```bash
python -m train.eval_dispatch_ppo \
  --config configs/final_main.yaml \
  --mask_self_check
```

Evaluation exports are written to `outputs/eval_exports/` as:

- per-episode CSV files
- JSON summary files
- optional Pareto sweep CSV and plots

## Config Files

The `configs/` directory contains scenario presets used by the newer evaluation pipeline:

- `final_small.yaml`
- `final_main.yaml`
- `final_large.yaml`
- `main_balanced.yaml`
- `main_noise_low.yaml`
- `main_noise_mid.yaml`
- `main_noise_high.yaml`
- `main_severe_mismatch.yaml`

These files define:

- simulation duration and seed
- synthetic grid size
- output directory root
- travel-time profile
- order generation settings
- rider generation settings

## Legacy Data Generation Path

If you want to regenerate the default `outputs/*.csv` dataset manually, the legacy generator modules can still be used:

```bash
python -m data_gen.module2_network
python -m data_gen.module3_travel_time
python -m data_gen.module4_orders
python -m data_gen.module5_riders
```

Important:

- these scripts currently expect a top-level `config.yaml`
- several older tests and utilities also still refer to `config.yaml`
- the newer `train.eval_dispatch_ppo` path is the more robust config-driven workflow

## ETA Model Training

Train the standalone edge ETA GNN:

```bash
python -m train.train_eta_gnn
```

Train the joint ETA head used by the node-set encoder:

```bash
python -m train.train_eta_head \
  --seed 0 \
  --steps 50000 \
  --out_dir outputs/eta_head \
  --eta_uncertainty
```

Evaluate a saved ETA-head checkpoint:

```bash
python -m train.eval_eta_head \
  --ckpt outputs/eta_head/eta_head_seed0_steps50000.pt \
  --episodes 50
```

## Joint Training And Analysis Utilities

Phase-1 style joint training:

```bash
python -m train.train_joint_phase1 \
  --seed 0 \
  --timesteps 50000 \
  --eta_coef 0.1 \
  --tb
```

Pareto sweep over ETA-risk coefficients:

```bash
python -m train.run_pareto_sweep \
  --model outputs/maskable_ueta_seed2.zip \
  --vecnorm outputs/vecnormalize.pkl \
  --episodes 200
```

Generate publication-style figures from a master experiment table:

```bash
python scripts/plot_master_experiment_figures.py \
  --input /path/to/master_experiment_table.csv \
  --out_dir outputs/paper_figures
```

Other useful helpers:

- `scripts/diagnose_policy.py`
- `scripts/inspect_obs.py`
- `scripts/profile_env_speed.py`
- `scripts/profile_env_split.py`
- `debug_route_test.py`

## Repository Layout

```text
.
|-- configs/
|-- data_gen/
|-- models/
|-- scripts/
|-- sim/
|-- train/
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Known Caveats

- The repository is in transition from a legacy single-file `config.yaml` flow to a newer `configs/*.yaml` flow.
- Training scripts are not fully unified. Some operate directly on the checked-in `outputs/` dataset, while the main evaluation script generates per-config datasets automatically.
- Checkpoint paths in commands are examples. If you train your own model, replace them with your run's actual output paths.
- Some analysis scripts assume local filesystem paths or pre-exported CSV tables and may need path edits before use.

## Suggested Starting Point

If you are new to the codebase, use this order:

1. Install dependencies.
2. Run `python -m train.train_dispatch_ppo --seed 0 --timesteps 2048`.
3. Run `python -m train.eval_dispatch_ppo --config configs/final_main.yaml --greedy_only --episodes 5`.
4. Train `train.train_dispatch_maskable_ppo`.
5. Compare learned policy vs. greedy/random baselines with `train.eval_dispatch_ppo` and `train.eval_dispatch_random`.

That sequence exercises the main simulator, policy pipeline, and evaluation/export flow with the least friction.
