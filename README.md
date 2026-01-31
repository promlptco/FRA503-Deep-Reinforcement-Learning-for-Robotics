# FRA503 – Deep Reinforcement Learning for Robotics  
## Homework 0: Cartpole Reinforcement Learning

**Authors**
- Chantouch Orungrote (66340500011)  
- Sasish Keawsing (66340500076)

---

## Objective

This homework studies the **Isaac-Cartpole-v0** reinforcement learning task to:
- Understand environment structure and RL components
- Analyze action space, observation space, rewards, and termination
- Experiment with **reward weight tuning**
- Relate practical results to **reinforcement learning fundamentals**

---

## Environment

**Task:** Isaac-Cartpole-v0  
**Config file:**
```

IsaacLab/source/isaaclab_tasks/isaaclab_tasks/
manager_based/classic/cartpole/cartpole_env_cfg.py

```

### Action Space
- Continuous (1D)
- Horizontal force applied to the cart joint (`joint_effort`)

### Observation Space
- 4D continuous state:
  - Cart position (x)
  - Cart velocity (ẋ)
  - Pole angle (θ)
  - Pole angular velocity (θ̇)

### Termination Conditions
- Episode timeout
- Cart position outside `[-3.0, 3.0]`

### Reward Terms
- Alive reward (+1.0)
- Termination penalty (-2.0)
- Pole position penalty (-1.0)
- Cart velocity penalty (-0.01)
- Pole angular velocity penalty (-0.005)

---

## Experiments

**Baseline model:** 448,000 training steps  
**Observation horizon:** 300 timesteps  

Five reward-weight experiments were conducted:
1. Alive reward
2. Termination penalty
3. Pole position penalty
4. Cart velocity penalty
5. Pole angular velocity penalty

Each reward was tested at:
- `0.0` (removed)
- Default value
- `10×` default

**Key Result:**  
Agent behavior changes directly with reward design, confirming the **Reward Hypothesis**.

---

## Reinforcement Learning Fundamentals

- **Agent:** Cart controller  
- **State:** Cart and pole positions & velocities  
- **Action:** Force applied to cart  
- **Reward:** Scalar feedback  
- **Policy:** \( \pi(s) \rightarrow a \)  
- **Value Function:** Expected return from a state  

---

## Report

Full results, plots, and analysis are available in:

```

report/HW0_DRL_6611_6676.pdf

```

---

## License

Educational use only (FRA503 coursework)
```

---

If you want an **even more minimal (submission-only) version** or one that **exactly matches a TA checklist**, I can compress it further.
