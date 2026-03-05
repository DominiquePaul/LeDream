# Gymnasium Quickstart with PPO

A minimal but complete tutorial for getting started with Gymnasium and training reinforcement learning agents using PPO (Proximal Policy Optimization).

## Features

- **Comprehensive Tutorial**: Jupyter notebook with step-by-step explanations
- **Easy Visualization**: Multiple visualization functions for observing agent behavior
- **Standalone Script**: Python script for quick training and testing
- **PPO Implementation**: Using Stable-Baselines3 for state-of-the-art RL
- **Side-by-Side Comparisons**: Compare random vs trained agents
- **Video Export**: Save agent performance as video files

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

The notebook provides an interactive learning experience with detailed explanations:

```bash
jupyter notebook gymnasium_quickstart.ipynb
```

Then run all cells to see the complete workflow from basics to trained agent visualization.

### Option 2: Standalone Script

For quick training and testing without notebooks:

```bash
# Train a new agent
python quickstart_ppo.py --train

# Test the trained agent
python quickstart_ppo.py --test

# Compare random vs trained
python quickstart_ppo.py --compare

# Create a video
python quickstart_ppo.py --video

# Train on different environment with more timesteps
python quickstart_ppo.py --train --env LunarLander-v3 --timesteps 100000
```

## What You'll Learn

1. **Gymnasium Basics**
   - Creating environments
   - Understanding observation and action spaces
   - Episode lifecycle (reset, step, done)

2. **Baseline Performance**
   - Testing random agents
   - Establishing performance baselines

3. **PPO Training**
   - Setting up PPO with optimal hyperparameters
   - Training loop with progress monitoring
   - Understanding PPO components (policy, value function, clipping)

4. **Visualization**
   - Real-time episode rendering
   - Animated visualizations in notebooks
   - Video export for sharing results
   - Training progress plots

5. **Evaluation**
   - Performance metrics (mean reward, std)
   - Comparing different agents
   - Statistical evaluation over multiple episodes

## Tutorial Structure

### Jupyter Notebook Sections

1. **Setup & Imports** - Get everything ready
2. **Gymnasium Basics** - Learn the interface
3. **Environment Exploration** - Understand CartPole
4. **Visualization Functions** - Reusable helper functions
5. **Random Baseline** - See untrained performance
6. **Training Callback** - Monitor training progress
7. **PPO Training** - Train the agent
8. **Training Progress** - Visualize learning curves
9. **Evaluation** - Test trained agent
10. **Visualization** - See the agent in action
11. **Comparison** - Random vs Trained side-by-side
12. **Video Export** - Save videos
13. **Model Persistence** - Save/load models
14. **Other Environments** - Try different tasks

## Key Features

### 🎥 Easy Visualization

The tutorial includes several visualization utilities:

```python
# Render and display an episode
frames, reward, steps = render_episode(env, agent=model)
display(display_frames(frames, title="Trained Agent"))

# Save as video
save_video(frames, 'demo.mp4', fps=30)

# Compare multiple agents
compare_agents(env, {
    'Random': None,
    'Trained PPO': model
})
```

### 📊 Training Monitoring

Real-time training progress with custom callbacks:

```python
callback = ProgressCallback(check_freq=2000)
model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
```

Shows:
- Steps completed
- Episodes completed
- Mean reward over last 100 episodes

### 🎯 Complete Workflow

```
Random Baseline → Train PPO → Evaluate → Visualize → Compare
```

Every step includes visualization to help you understand what's happening.

## Environments to Try

The tutorial uses CartPole-v1 by default (perfect for learning), but you can easily try:

- **Acrobot-v1** - Swing up a two-link robot arm
- **MountainCar-v0** - Drive up a steep hill with momentum
- **LunarLander-v3** - Land a spacecraft safely
- **Pendulum-v1** - Balance an inverted pendulum (continuous control)

Just change the environment name in the notebook or use the `--env` flag in the script.

## Customization

### Hyperparameters

The PPO agent uses well-tested defaults, but you can tune:

```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,    # Step size for gradient descent
    n_steps=2048,          # Steps collected before each update
    batch_size=64,         # Minibatch size for optimization
    n_epochs=10,           # Number of epochs per update
    gamma=0.99,            # Discount factor
    gae_lambda=0.95,       # GAE parameter for advantage estimation
    clip_range=0.2,        # PPO clipping parameter
)
```

### Training Duration

- **CartPole**: 50,000 steps (5-10 minutes)
- **LunarLander**: 100,000-200,000 steps (20-40 minutes)
- **Harder environments**: May need 500,000+ steps

## Output Files

The tutorial generates:

- `ppo_cartpole.zip` - Saved model
- `training_progress.png` - Training curves
- `agent_comparison.png` - Performance comparison
- `ppo_cartpole_demo.mp4` - Video demonstration

## Understanding PPO

PPO (Proximal Policy Optimization) is one of the most popular RL algorithms because:

1. **Stable**: Uses clipped objective to prevent destructive policy updates
2. **Sample Efficient**: Reuses collected experience multiple times
3. **Easy to Tune**: Robust default hyperparameters work well
4. **General**: Works for both discrete and continuous control

The tutorial shows how PPO learns by:
- Collecting experience through environment interaction
- Computing advantages (how good actions were)
- Updating policy to increase probability of good actions
- Limiting update size to maintain stability

## Tips for Success

1. **Start Simple**: Master CartPole before moving to harder environments
2. **Visualize Often**: Watch your agent to understand its behavior
3. **Compare Baselines**: Always compare to random agents
4. **Save Models**: Don't lose your trained agents!
5. **Monitor Training**: Use callbacks to track progress
6. **Be Patient**: Some environments need lots of training

## Common Issues

**Agent not learning?**
- Increase training timesteps
- Check if environment is too hard
- Try adjusting learning rate

**Training too slow?**
- Use fewer timesteps for initial experiments
- Try simpler environments first
- Consider using a GPU (for image-based environments)

**Visualization not working?**
- Make sure `render_mode='rgb_array'` is set
- Check that imageio and matplotlib are installed

## Next Steps

After completing this tutorial:

1. **Try Different Environments**
   - Start with Acrobot-v1 or MountainCar-v0
   - Progress to LunarLander-v3
   - Eventually try MuJoCo environments

2. **Experiment with Algorithms**
   - A2C (faster but less stable than PPO)
   - SAC (for continuous control)
   - TD3 (another continuous control algorithm)

3. **Build Custom Environments**
   - Create your own Gymnasium environment
   - Define custom observation/action spaces
   - Implement your own reward function

4. **Advanced Topics**
   - Hyperparameter tuning
   - Curriculum learning
   - Multi-agent RL
   - Transfer learning

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

## Requirements

All dependencies are already installed in your environment:
- gymnasium[mujoco]
- stable-baselines3[extra]
- imageio
- matplotlib
- numpy
- tqdm

## License

This tutorial is provided as-is for educational purposes.

## Contributing

Feel free to extend this tutorial with:
- Additional environments
- More visualization types
- Advanced PPO configurations
- Comparison with other algorithms

---

**Happy Learning! 🚀**

For questions or issues, refer to the inline comments in the notebook or script.
