# 🚀 Gymnasium + PPO Quick Start Guide

Get started with reinforcement learning in minutes!

## ⚡ 2-Minute Demo

The fastest way to see RL in action:

```bash
cd gymnasium_tutorial
uv run python demo_visual.py
```

This will:
1. Create a CartPole environment
2. Test a random agent (baseline)
3. Train a PPO agent (~60 seconds)
4. Compare before/after performance
5. Generate videos and plots

**Output:**
- `random_agent.mp4` - Untrained agent
- `trained_agent.mp4` - Trained agent
- `comparison.mp4` - Side-by-side comparison
- `performance_comparison.png` - Performance chart

---

## 📚 Interactive Tutorial

For a complete learning experience:

```bash
uv run jupyter notebook gymnasium_quickstart.ipynb
```

**What's inside:**
- 14 sections covering everything from basics to advanced
- Step-by-step explanations
- Interactive visualizations
- Reusable code snippets
- Multiple example environments

**Time:** 30-45 minutes to complete

---

## 💻 Command Line Interface

For quick experiments without notebooks:

### Train an agent
```bash
uv run python quickstart_ppo.py --train
```

Options:
- `--env CartPole-v1` - Environment name (default: CartPole-v1)
- `--timesteps 50000` - Training steps (default: 50000)
- `--model ppo_model` - Save path (default: ppo_model)

### Test trained agent
```bash
uv run python quickstart_ppo.py --test
```

Evaluates the agent over multiple episodes and shows statistics.

### Compare agents
```bash
uv run python quickstart_ppo.py --compare
```

Compares random vs trained agent with statistical analysis and plots.

### Create video
```bash
uv run python quickstart_ppo.py --video
```

Saves a video demonstration of the trained agent.

### Combined example
```bash
# Train, test, compare, and create video all at once
uv run python quickstart_ppo.py --train --test --compare --video
```

### Try different environments
```bash
# Train on LunarLander
uv run python quickstart_ppo.py --train --env LunarLander-v3 --timesteps 100000

# Train on Acrobot
uv run python quickstart_ppo.py --train --env Acrobot-v1 --timesteps 75000
```

---

## 🎯 Choose Your Path

| Path | Time | Depth | Best For |
|------|------|-------|----------|
| **Demo** | 2 min | Quick overview | First impressions, demos |
| **Notebook** | 30-45 min | Comprehensive | Learning, understanding |
| **CLI** | 5-10 min | Practical | Experiments, research |

---

## 📖 Learning Objectives

After completing this tutorial, you'll understand:

### Fundamentals
- ✓ What is Gymnasium and how to use it
- ✓ Observation spaces and action spaces
- ✓ Episode lifecycle (reset, step, done)
- ✓ Reward functions and cumulative returns

### PPO Algorithm
- ✓ Why PPO is popular (stable, sample-efficient, easy)
- ✓ Key components (policy, value function, clipping)
- ✓ Hyperparameters and their effects
- ✓ Training loop and optimization

### Practical Skills
- ✓ Setting up environments
- ✓ Training and evaluating agents
- ✓ Visualizing agent behavior
- ✓ Comparing agent performance
- ✓ Saving and loading models
- ✓ Creating videos and plots

### Best Practices
- ✓ Establishing baselines (random agent)
- ✓ Monitoring training progress
- ✓ Statistical evaluation (mean, std, multiple seeds)
- ✓ Visualization techniques
- ✓ Model persistence

---

## 🎮 Environments to Try

### Beginner (Discrete Actions)
- **CartPole-v1** ⭐ Start here!
  - Goal: Balance pole on cart
  - Actions: Push left/right
  - Training: ~50,000 steps

- **Acrobot-v1**
  - Goal: Swing up double pendulum
  - Actions: Torque on middle joint
  - Training: ~75,000 steps

### Intermediate
- **LunarLander-v3** ⭐ Beautiful visuals!
  - Goal: Land spacecraft safely
  - Actions: Fire thrusters
  - Training: ~100,000 steps

- **MountainCar-v0**
  - Goal: Drive up steep hill
  - Actions: Accelerate left/right
  - Training: ~100,000 steps

### Advanced (Continuous Actions)
- **Pendulum-v1**
  - Goal: Balance inverted pendulum
  - Actions: Continuous torque
  - Algorithm: Use SAC or TD3 instead of PPO

- **MuJoCo Environments**
  - See `mujoco_tutorial/` for physics-based control

---

## 🛠️ Troubleshooting

### Import errors
```bash
# Reinstall dependencies
uv sync

# Verify setup
uv run python test_setup.py
```

### Slow training
- Reduce `--timesteps` for faster experiments
- CartPole is fastest (50k steps ~1-2 minutes)
- LunarLander takes longer (100k steps ~5-10 minutes)

### Agent not learning
- Check if environment is too hard
- Increase training timesteps
- Try adjusting learning rate (3e-4 is default)
- Verify reward function makes sense

### Visualization not working
- Ensure `render_mode='rgb_array'` is set
- Check matplotlib backend: `import matplotlib; matplotlib.use('TkAgg')`
- Try saving video instead of live display

---

## 🔥 Quick Tips

1. **Always start with CartPole** - It's fast and easy to debug
2. **Watch your agent** - Visualization helps debug issues
3. **Compare to baseline** - Always test a random agent first
4. **Save your models** - Training takes time!
5. **Monitor training** - Use progress callbacks
6. **Multiple seeds** - Test with different random seeds
7. **Start simple** - Master basics before moving to complex envs

---

## 📈 Expected Results

### CartPole-v1 (50k steps)
- Random agent: ~20-30 reward
- Trained PPO: ~450-500 reward ✅
- Training time: 1-2 minutes

### LunarLander-v3 (100k steps)
- Random agent: ~-150 to -200 reward
- Trained PPO: ~200-250 reward ✅
- Training time: 5-10 minutes

### Acrobot-v1 (75k steps)
- Random agent: ~-450 to -500 reward
- Trained PPO: ~-80 to -100 reward ✅
- Training time: 3-5 minutes

---

## 📚 Next Steps

After mastering this tutorial:

1. **Experiment with hyperparameters**
   - Learning rate: 1e-4 to 1e-3
   - Batch size: 32, 64, 128
   - Network size: Change policy architecture

2. **Try other algorithms**
   - A2C (faster, less stable)
   - SAC (continuous control)
   - TD3 (continuous control)

3. **Build custom environments**
   - Create your own Gymnasium environment
   - Define custom observation/action spaces
   - Implement custom reward functions

4. **Advanced topics**
   - Hyperparameter tuning with Optuna
   - Curriculum learning
   - Domain randomization
   - Transfer learning

5. **Move to MuJoCo**
   - See `mujoco_tutorial/` for physics simulation
   - Complex robots and dynamics
   - Realistic control problems

---

## 🎓 Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Deep RL Course by Hugging Face](https://huggingface.co/deep-rl-course)

---

## ❓ FAQ

**Q: How long does training take?**
A: CartPole: 1-2 min, LunarLander: 5-10 min (on CPU)

**Q: Can I use GPU?**
A: PPO with MLP policies doesn't benefit much from GPU. Use GPU for vision-based tasks.

**Q: Which algorithm should I use?**
A: PPO is great for discrete actions. For continuous, try SAC or TD3.

**Q: Why isn't my agent learning?**
A: Check: (1) Environment is appropriate, (2) Enough training steps, (3) Reward function, (4) Hyperparameters

**Q: How do I create custom environments?**
A: Subclass `gymnasium.Env` and implement `reset()`, `step()`, observation/action spaces.

**Q: Can I train multiple agents?**
A: Yes! Use vectorized environments with `stable_baselines3.common.vec_env`

---

## 🤝 Contributing

Found a bug or have a suggestion? The tutorial is designed to be extended!

Ideas for extensions:
- Additional environments
- More visualization types
- Hyperparameter tuning examples
- Custom environment templates
- Multi-agent scenarios

---

**Ready to start? Pick your path above and dive in! 🚀**
