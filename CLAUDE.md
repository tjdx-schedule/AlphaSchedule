# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaSchedule applies AlphaZero-style reinforcement learning methods to scheduling problems, specifically job-shop scheduling with deadlines and priorities. The repository contains multiple algorithm implementations and versions exploring different approaches to the scheduling optimization problem.

## Problem Domain

The core scheduling problem:
- **Jobs**: Each has processing time, deadline, and priority
- **Machines**: Multiple machines can process jobs in parallel
- **Objective**: Minimize weighted tardiness (priority × lateness for late jobs)
- **Environment**: Jobs are assigned to machines sequentially; the scheduler chooses which job to schedule next

## Repository Structure

### Algorithm Implementations

The repository contains several parallel implementations organized by approach:

1. **MCTS + Policy Network** (`mctsAlphaV0.0XX/`): AlphaZero-style approach combining Monte Carlo Tree Search with neural network policy guidance
   - Latest version: `mctsAlphaV0.078/`
   - Uses beam search with MCTS
   - Optional simulated annealing post-processing
   - Policy-value network based on ResNet architecture

2. **PPO** (`ppo_policyV0.0XX/`): Proximal Policy Optimization using actor-critic
   - Latest version: `ppo_policyV0.040/`
   - Vectorized parallel environments
   - Based on modified a2c_ppo_acktr implementation

3. **Genetic Algorithm** (`gaV0.021jobshop/`, `gaParallel/`): Traditional optimization baseline
   - Uses crossover and mutation operations on job sequences

4. **Other Baselines**: Simulated annealing (`saparallel/`), Particle Swarm Optimization (`psoParallel/`)

### Key Components Common Across Implementations

Each algorithm version typically contains:

- **Environment/Scheduler** (`scheduler.py`, `scheEnv.py`, `EnvirConf.py`):
  - `EnvirConf.py`: Configuration for problem size (partNum, machNum), difficulty distribution ('h'/'m'/'l'), and seed management
  - `Scheduler`: Core scheduling logic - assigns jobs to machines, tracks state, calculates objective
  - `ScheEnv`: Gym-compatible wrapper around Scheduler

- **Observation/State** (`obs.py`, `stateobs.py`):
  - Converts scheduler state into 2D feature maps for neural network input
  - Features: job characteristics (time, deadline, priority) and machine states

- **Policy/Model** (`policy_value_net_pytorch.py`, `mcts_policy.py`, `model.py`):
  - Neural network architectures for action selection
  - MCTS versions use ResNet-based policy-value networks
  - PPO uses separate policy and value heads

- **Logging** (`excel.py`, `ExcelLog.py`): Excel-based logging for experiments

- **Visualization** (`gantt.py`): Gantt chart generation for schedule visualization

## Development Workflow

### MCTS Versions (mctsAlphaV0.0XX)

**Dependencies:**
```bash
cd mctsAlphaV0.078
pip install -r requirements.txt
```

**Testing a trained policy:**
```bash
cd mctsAlphaV0.078
python testPolicy.py
```

**Key configuration:**
- Edit `venvs/EnvirConf.py` to change problem size and distribution
- Model weights stored in `models/` directory with naming pattern `{partNum}-{machNum}-weight.model`
- Test modes: `pure_policy`, `mcts_policy`, `mcts_policy_sa` (with simulated annealing)

**Architecture notes:**
- `Game` class manages interaction between environment and player
- `MCTSPlayer` implements beam search MCTS with optional SA refinement
- Supports parallel environment execution via `make_vec_envs` for efficient rollouts
- `venvs/baselines_com/vec_env/`: Custom vectorized environment wrappers for parallel execution

### PPO Versions (ppo_policyV0.0XX)

**Training:**
```bash
cd ppo_policyV0.040
python main.py --env-name ml-agent --num-processes 8 --num-steps 1024
```

**Evaluation:**
```bash
cd ppo_policyV0.040
python evaluation.py --load-dir ./trained_models/ppo --eval-num 100
```

**Key arguments (see `a2c_ppo_acktr/arguments.py`):**
- `--load`: Resume training from checkpoint
- `--transfer`: Transfer learning from pre-trained model
- `--excel-save`: Enable Excel logging
- `--eval-interval`: Evaluation frequency during training
- `--num-processes`: Number of parallel environments
- `--save-dir`: Model checkpoint directory

**Architecture notes:**
- `a2c_ppo_acktr/`: Contains PPO algorithm implementation, model, and storage
- `a2c_ppo_acktr/game/`: Environment wrappers and configuration
- Action masking prevents invalid actions (selecting already-scheduled jobs)
- Separate observation spaces for policy (2D feature map) and value (scalar)

### GA Versions (gaV0.021jobshop, gaParallel)

**Running:**
```bash
cd gaV0.021jobshop
python main.py
```

**Configuration:**
- Edit `config.py` for problem parameters
- GA operates on job sequence permutations
- `genpart.py`: Job generation and chromosome encoding

## Important Conventions

### Environment Modes
All implementations support three modes:
- `'train'`: Training instances with random seed incrementation
- `'test'`: Fixed test instances for evaluation
- `'val'`: Validation instances

Seed management in `EnvirConf.py`:
- `trainSeed`: Random for each run
- `testSeed`: Fixed at 0, increments per episode
- `valSeed`: Fixed at 1000, increments per episode

### State Representation
- Jobs and machines represented as matrices
- Job matrix: `[processing_time, deadline, priority]` per job
- Machine matrix: Current completion time per machine
- Observation: Multi-channel 2D feature map (width × height × features)

### Action Space
- Discrete: Select which job to schedule next (0 to partNum-1)
- Action masking ensures only unscheduled jobs are selectable
- Environment automatically assigns to earliest-available machine

### Objective Calculation
```python
grade = -sum(max(0, -tardiness) * priority for each job)
```
Lower grade is better (minimize weighted tardiness).

## Model Persistence

**MCTS models:**
- Stored as PyTorch state dicts
- Format: `{partNum}-{machNum}-weight.model`
- Contains: `com_base`, `policy_head`, `dist0_fc`, `value_head` components

**PPO models:**
- Stored using torch.save with agent, normalizer, and logger
- Format: `{env_name}.pt`
- Load with `torch.load()` and restore to agent

## Version Differences

- **V0.070**: Original MCTS implementation, simpler environment
- **V0.075ParallelProcess**: Added parallel environment rollouts
- **V0.077**: Refined MCTS with better beam search
- **V0.078**: Latest, cleaned up vectorized environments
- **ppo V0.030**: Earlier PPO version
- **ppo V0.040**: Current PPO with improved training stability

When working across versions, note that environment implementations (`scheEnv.py`, `scheduler.py`) may have slight API differences.

## Testing and Debugging

- Test scripts typically run multiple episodes and report min/mean/max grades
- Gantt chart visualization available via `plotGantt()` on completed schedules
- `past/` and `debug/` directories contain older/experimental code
- Log files (`logs.txt`, `log.txt`) contain training/evaluation history
