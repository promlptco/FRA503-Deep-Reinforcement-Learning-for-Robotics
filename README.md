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

[
[x,\ \theta,\ \dot{x},\ \dot{\theta}]
]

### Action

* Continuous horizontal force (scaled by 100.0)

### Termination

* Episode timeout
* Cart out of bounds: (|x| > 3.0)

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

## Reward Experiments

A baseline model trained for **448k steps** was evaluated with modified reward weights.

**Key observations:**

* Removing alive reward → immediate failure
* Strong pole position penalty → most stable control
* Excessive penalties → conservative or unstable behavior

---

## RL Mapping

* **Agent:** Policy network
* **State:** ((x,\ \dot{x},\ \theta,\ \dot{\theta}))
* **Action:** Cart force
* **Reward:** Survival + stability penalties

[
\pi(s) \rightarrow a,\quad
V^\pi(s) = \mathbb{E}[G_t \mid s,\pi]
]

---

## Takeaway

Balanced reward shaping—especially pole position control—is critical for stable learning in continuous control tasks.

---

If you want, I can:

* Add **training curves / gifs**
* Make this **even shorter**
* Write a **one-paragraph project description**
* Align it with **course submission guidelines**
