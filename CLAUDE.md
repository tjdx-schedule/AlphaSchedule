# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaSchedule applies AlphaZero-style methods to job-shop scheduling optimization. The goal is to minimize weighted tardiness by assigning jobs to machines. Multiple algorithm implementations are provided for comparison:

- **MCTS with Policy-Value Networks** (`mctsAlphaV0.077/`, `mctsAlphaV0.078/`) - AlphaZero-style tree search
- **PPO (Proximal Policy Optimization)** (`sci1-ppoV0.040/`, `ppo_policyV0.040/`) - Reinforcement learning
- **Genetic Algorithm** (`gaV0.021jobshop/`, `gaParallel/`) - Evolutionary optimization
- **Simulated Annealing** (`saparallel/`) - Metaheuristic search
- **PSO (Particle Swarm Optimization)** (`psoParallel/`) - Swarm intelligence

## Development Commands

Use `uv` to manage Python and run scripts.

### PPO Training
```bash
cd sci1-ppoV0.040/
uv run python main.py --env-name ScheduleEnv --part-num 65 --dist-type h --num-processes 8 --run-hours 24
```

### PPO Evaluation
```bash
cd sci1-ppoV0.040/
uv run python enjoy.py --env-name ScheduleEnv --part-num 65 --dist-type h --load-dir ./trained_models/
```

### MCTS Testing
```bash
cd mctsAlphaV0.077/
uv run python testPolicy.py
```

### Genetic Algorithm
```bash
cd gaV0.021jobshop/
uv run python main.py
```

### Install Dependencies
```bash
uv pip install -r requirements.txt
```

## Core Architecture

### Scheduling Environment

All algorithms share a common scheduling abstraction:

- **`scheduler.py`** - Core `Scheduler` class managing job-machine assignment state
- **`EnvirConf.py`** - `EnviroConfig` class defining problem parameters
- **`genpart.py`** - `GenPart` class generating job instances with reproducible seeds

### Environment Configuration (`EnvirConf.py`)

Key parameters in `EnviroConfig`:
- `partNum` - Number of jobs (default: 65)
- `machNum` - Number of machines (derived: `(partNum - 5) // 10 * 5`)
- `distType` - Distribution tightness: 'h' (high), 'm' (medium), 'l' (low)
- Seeds: `trainSeed` (random), `testSeed` (0), `valSeed` (1000) for reproducibility

Distribution parameters by type:
```python
'h': [tight=0.3, priority=20, maxTime=200]
'm': [tight=0.5, priority=12, maxTime=125]
'l': [tight=0.65, priority=6, maxTime=50]
```

### Job Representation

Jobs are numpy arrays with columns:
- Column 0: Processing time
- Column 1: Due date (deadline)
- Column 2: Priority weight

### Action Space

Actions select which job to schedule next. The scheduler assigns it to the machine with minimum current load (earliest availability).

### Reward/Grade Calculation

- **Grade** = sum of (tardiness × priority) for late jobs
- Lower grade = better solution
- Tardiness = max(0, completion_time - due_date)

## PPO Implementation (`sci1-ppoV0.040/`)

### Key Files
- `main.py` - Training loop with parallel environments
- `a2c_ppo_acktr/model.py` - Policy network architecture
- `a2c_ppo_acktr/algo/ppo.py` - PPO update algorithm
- `a2c_ppo_acktr/storage.py` - Rollout buffer
- `a2c_ppo_acktr/arguments.py` - Hyperparameters
- `a2c_ppo_acktr/game/envs.py` - Vectorized environment wrapper
- `a2c_ppo_acktr/game/scheEnv.py` - Gym-compatible scheduling environment

### Key Hyperparameters
- `--num-processes 8` - Parallel environment count
- `--num-steps 1024` - Steps per rollout
- `--ppo-epoch 4` - Update epochs per batch
- `--clip-param 0.1` - PPO clipping threshold
- `--lr 2.5e-4` - Learning rate
- `--gamma 1.0` - Discount factor (undiscounted)

## MCTS Implementation (`mctsAlphaV0.077/`)

### Key Files
- `testPolicy.py` - Main execution script
- `mcts_policy.py` - `MCTS` and `MCTSPlayer` classes with UCB selection
- `policy_value_net_pytorch.py` - Neural network for policy/value prediction
- `venvs/game.py` - Game wrapper for MCTS interface

### Search Parameters
- `beam_size` - Number of nodes to expand per level
- Dirichlet noise (`EPSILON=0.4`, `DIRNOISE=0.1`) for exploration at root

## Genetic Algorithm (`gaV0.021jobshop/`)

### Key Files
- `main.py` - Entry point
- `ga.py` - `GeneAlgorithm` and `GaOpeartor` classes
- `genpart.py` - Job generation and `Scheduler` class

### GA Parameters
- `popuSize` - Population size (default: 50)
- `mutateRate` - Mutation probability (default: 0.2)
- `gap` - Generation gap for crossover (default: 0.75)

## Logging

Results are logged to Excel files via `ExcelLog` class (in `excel.py` or `ExcelLog.py`). Training metrics, test grades, and loss values are tracked.

## Common Patterns

### Mode-based Seed Management
All algorithms use `mode` parameter ('train', 'test', 'val') to select appropriate random seeds for reproducibility across experiments.

### Plotting
Gantt chart visualization available via `gantt.py` / `plotGantt()` function.
