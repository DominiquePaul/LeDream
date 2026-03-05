# RL Tutorials Collection

Comprehensive tutorials for reinforcement learning, from basics to advanced physics simulation.

## 🚀 Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create venv
uv sync

# Register the kernel (first time only)
uv run python -m ipykernel install --user --name=mujoco-tutorial --display-name="MuJoCo Tutorial (uv)"
```

## 📚 Tutorials

### 1. Gymnasium Quickstart with PPO
**Path:** `gymnasium_tutorial/`

The perfect starting point for RL beginners! Learn Gymnasium basics and train your first PPO agent with easy-to-use visualization.

**Quick Demo (2 minutes):**
```bash
cd gymnasium_tutorial
uv run python demo_visual.py
```

**Full Tutorial:**
```bash
uv run jupyter notebook gymnasium_tutorial/gymnasium_quickstart.ipynb
```

**Command Line Training:**
```bash
# Train a PPO agent
uv run python gymnasium_tutorial/quickstart_ppo.py --train

# Test and visualize
uv run python gymnasium_tutorial/quickstart_ppo.py --test --compare --video
```

**What you'll learn:**
- Gymnasium environment basics
- PPO algorithm fundamentals
- Easy visualization techniques
- Training and evaluation workflows
- Video export and comparisons

### 2. MuJoCo Tutorial
**Path:** `mujoco_tutorial/`

Deep dive into advanced physics simulation and complex RL environments.

**Run the tutorial:**
```bash
uv run jupyter notebook mujoco_tutorial/mujoco_tutorial.ipynb
# In Jupyter: Kernel → Change Kernel → "MuJoCo Tutorial (uv)"
```

**What's inside:**
- **`mujoco_tutorial.ipynb`** -- comprehensive notebook covering 9 sections:
  1. MuJoCo fundamentals (`mjModel` vs `mjData`)
  2. MJCF modeling language (building robots in XML)
  3. Simulation loop (forward dynamics, time stepping, chaos)
  4. Actuators & control (motor vs position, swing-up demo)
  5. Sensors & observations (building obs vectors for RL)
  6. Contact physics (collisions, mass matrix, dynamics)
  7. MuJoCo vs Isaac Sim/Lab (comparison & when to use which)
  8. Gymnasium integration (standard RL env API)
  9. Training a PPO agent on InvertedPendulum

- **`models/`** -- custom MJCF models with annotated XML:
  - `simple_pendulum.xml` -- 1-DOF hello world
  - `cart_pole.xml` -- under-actuated classic control
  - `double_pendulum.xml` -- chaotic dynamics
  - `reacher.xml` -- 2-link robotic arm
  - `contact_scene.xml` -- collision & contact physics

## 🎯 Learning Path

**Beginner:** Start with the Gymnasium tutorial
1. `gymnasium_tutorial/demo_visual.py` - Quick 2-minute demo
2. `gymnasium_tutorial/gymnasium_quickstart.ipynb` - Complete beginner tutorial
3. Experiment with different environments (CartPole, LunarLander)

**Intermediate:** Move to MuJoCo
1. `mujoco_tutorial/mujoco_tutorial.ipynb` - Physics simulation deep dive
2. Build custom environments
3. Train on complex control tasks

**Advanced:** Combine both
1. Create custom MuJoCo environments
2. Apply advanced RL algorithms
3. Implement curriculum learning and domain randomization

## 📊 Features

All tutorials include:
- ✓ Interactive Jupyter notebooks
- ✓ Standalone Python scripts
- ✓ Easy-to-use visualization functions
- ✓ Real-time training progress monitoring
- ✓ Video export capabilities
- ✓ Performance comparison tools
- ✓ Model saving/loading
- ✓ Comprehensive documentation

## 🛠️ Requirements

All dependencies are managed via `pyproject.toml`:
- gymnasium[mujoco] - RL environment interface
- stable-baselines3[extra] - State-of-the-art RL algorithms
- mujoco - Advanced physics simulation
- matplotlib - Plotting and visualization
- imageio - Video creation
- numpy, tqdm - Utilities
