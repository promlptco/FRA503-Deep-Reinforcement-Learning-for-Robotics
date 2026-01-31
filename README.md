# Homework 1: Multi-Armed Bandit

### Author
- Chantouch Orungrote (6340500011)
- Sasish Kaewsing (6340500076)

---

### Part 1: Setting up Multi-armed Bandit

We implemented a complete multi-armed bandit framework from scratch as explained in class.

Components include:

1. **Bandit class** (`bandit.py`)
   - Constructor initializes `n_arms` with hidden reward distribution
     - Supports **Bernoulli** (fixed success probabilities) and **Gaussian** (random means per run)
   - `pull(arm)` function returns stochastic reward

2. **Agent class** (`agent.py`)
   - Constructor initializes learnable parameters (`Q` values, `N` counts) and strategy
   - `act()` selects action (ε-greedy or UCB)
   - `update(action, reward)` updates Q-values using incremental sample-average method

3. **Simulation script** (`main.py`)
   - Runs multiple independent experiments
   - Averages results over `N_RUNS`
   - Computes average reward, % optimal action, cumulative regret
   - Detects convergence time
   - Generates required plots

### Part 2: Implementing epsilon-greedy algorithm

- Implemented in `agent.py` (strategy="eps")
- Tested with multiple ε values: 0.0, 0.01, 0.05, 0.1, 0.5
- Plots action selection percentage over time for each arm (timesteps vs reward for each bandit)
- Convergence analysis performed (see Part 5)

### Part 3: Implementing UCB

- Implemented UCB1 in `agent.py` (strategy="ucb")
- Tested with multiple c values: 0.5, 1.0, 1.5, 2.0, 5.0
- Plots action selection percentage over time for each arm
- Convergence and regret analysis included

## Installation

```bash
# Recommended: use virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install numpy matplotlib
```

## How to Run

```bash
python main.py
```

- Results are saved in `figures/bernoulli/` and `figures/gaussian/`
- Console shows progress and convergence time for each configuration

## Global Parameters (config.py)

| Parameter         | Default | Description                                      |
|-------------------|---------|--------------------------------------------------|
| N_ARMS            | 10      | Number of arms                                   |
| N_RUNS            | 3000    | Number of independent runs (for averaging)       |
| N_STEPS           | 600     | Timesteps per experiment                         |
| REWARD_TYPES      | ["bernoulli", "gaussian"] | Bandit reward distributions to compare |
| EPSILON_VALS      | [0.0, 0.01, 0.05, 0.1, 0.5] | ε-greedy exploration rates |
| UCB_C_VALS        | [0.5, 1.0, 1.5, 2.0, 5.0] | UCB exploration constants |

**N_RUNS vs N_STEPS**  
- N_STEPS: number of arm pulls per single experiment  
- N_RUNS: number of independent repetitions (used for averaging to reduce variance)

## Generated Outputs

All results are separated by reward type:

```
figures/
├── bernoulli/
│   ├── average_reward.png
│   ├── percent_optimal.png
│   ├── cumulative_regret.png
│   ├── action_selection_eps_best.png
│   └── action_selection_ucb_best.png
└── gaussian/
    └── (same files)
```

## Convergence Analysis

Convergence is defined as:

> The earliest timestep t where the 50-step rolling average of optimal action percentage ≥ 85% **and** remains ≥ 85% until the end of the experiment.

Typical results (example from runs):

**Bernoulli setting** (fixed probabilities, optimal arm p=0.80)  
- ε=0.1: converges around **~220–300 steps**  
- UCB c=2.0: converges around **~120–180 steps** (faster and more stable)

**Gaussian setting** (random means per run)  
- ε=0.1: converges around **~150–250 steps**  
- UCB c=2.0: converges around **~80–140 steps**

UCB generally converges faster and achieves higher final % optimal action than ε-greedy.

## Project Structure

```
├── bandit.py      # Bandit environment (Bernoulli & Gaussian)
├── agent.py       # ε-greedy and UCB agent implementation
├── config.py      # Hyperparameters and experiment configurations
├── plot.py        # All plotting functions
├── main.py        # Main simulation loop and result generation
└── figures/       # Generated plots (bernoulli/ & gaussian/)
```
