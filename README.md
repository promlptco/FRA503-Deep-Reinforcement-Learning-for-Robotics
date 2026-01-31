# Homework 0: CartPole Exploration

**Authors**
- Chantouch Orungrote (66340500011)
- Sasish Keawsing (66340500076)

---

## Overview

This project explores the **Isaac-CartPole-v0** environment using reinforcement learning. We analyze the default setup, study the impact of reward shaping, and relate the implementation to core RL concepts.

---

## Environment

* **Simulator:** Isaac Sim
* **Task:** Isaac-CartPole-v0
* **Config:**
  `isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py`

### State

[cart_pos, cart_vel, pole_pos, pole_vel]

### Action

- Continuous horizontal force (scaled by 100.0)

### Termination

- Episode timeout
- Cart out of bounds: (|x| > 3.0)

---

## Reward Function

| Term                          | Weight |
| ----------------------------- | ------ |
| Alive reward                  | +1.0   |
| Termination penalty           | −2.0   |
| Pole position penalty         | −1.0   |
| Cart velocity penalty         | −0.01  |
| Pole angular velocity penalty | −0.005 |

---
