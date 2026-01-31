# Homework 0: Cartpole Exploration

**Authors:** 
- Chantouch Orungrote (66340500011)
- Sasish Keawsing (66340500076)

## Overview
Exploration of Isaac-Cartpole-v0 environment using reinforcement learning. Part 1 examines default setup, Part 2 experiments with reward weights, Part 3 maps RL fundamentals.

---

## Part 1: Cartpole RL Agent

### Training Setup
- Environment: Isaac-Cartpole-v0 in Isaac Sim
- Configuration file: `isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py`

### Configurations
- **Action Space:** Continuous force on cart (scaled by 100.0)
- **Observation Space:** 4D vector [`cart pos, pole pos, cart vel, pole vel`]
- **Termination:** Timeout or cart out of bounds (|x| > 3.0)
- **Rewards (5 terms):**
  - +1.0 Alive reward
  - -2.0 Termination penalty
  - -1.0 Pole position penalty
  - -0.01 Cart velocity penalty
  - -0.005 Pole angular velocity penalty

---

## Part 2: Reward Weight Experiments

Baseline model (448k steps) tested with individual weight modifications (0.0 or ×10).

### Results

| Experiment | Weight Adjust | Behavior |
|------------|---------------|--------------|
| Alive Reward | 0.0 / +10.0 | 0.0 → immediate failure; +10.0 → unstable survival-focused policy |
| Termination Penalty | 0.0 / -20.0 | 0.0 → less refined; -20.0 → overly conservative |
| Pole Position | 0.0 / -10.0 | 0.0 → chaotic oscillations; -10.0 → most stable control |
| Cart Velocity | 0.0 / -0.1 | 0.0 → aggressive moves; -0.1 → overly constrained |
| Pole Angular Velocity | 0.0 / -0.05 | 0.0 → noisy oscillations; -0.05 → slow, damped response |

---

## Part 3: RL Fundamentals

### Core Components
- **Agent:** Policy network controlling cart force
- **State:** (x, ẋ, θ, θ̇)
- **Action:** Horizontal force
- **Reward:** Positive for upright pole, penalties for deviation/termination

### Key Concepts

| Concept | Definition | Scope |
|---------|------------|-------|
| Reward (R) | Immediate feedback | Single timestep |
| Return (G) | Cumulative discounted reward | Episode |
| Value (V) | Expected future return | Long-term |

### Mathematical Functions

- **Policy:** π(s) → a
- **Value:** V^π(s) = E[G_t | s, π]
- **Transition:** P(s' | s, a)
- **Reward Model:** R(s, a)
