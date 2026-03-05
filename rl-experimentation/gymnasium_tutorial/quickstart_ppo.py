"""
Gymnasium Quickstart with PPO - Standalone Script

This script provides a minimal but complete example of training a PPO agent
with easy visualization capabilities.

Usage:
    python quickstart_ppo.py --train          # Train a new agent
    python quickstart_ppo.py --test           # Test saved agent
    python quickstart_ppo.py --compare        # Compare random vs trained
    python quickstart_ppo.py --video          # Create video of trained agent
"""

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')


class ProgressCallback(BaseCallback):
    """Custom callback for displaying training progress."""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.current_reward = 0
        
    def _on_step(self):
        # Track reward
        self.current_reward += self.locals['rewards'][0]
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        
        # Periodic logging
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards)
                print(f"Steps: {self.num_timesteps:6d} | "
                      f"Episodes: {len(self.episode_rewards):4d} | "
                      f"Mean Reward (last 100): {mean_reward:.2f}")
        
        return True


def render_episode(env, agent=None, max_steps=500, seed=None, render=True):
    """
    Run a single episode and optionally render it.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent (if None, uses random actions)
        max_steps: Maximum steps per episode
        seed: Random seed
        render: Whether to collect frames
    
    Returns:
        frames, total_reward, steps
    """
    frames = [] if render else None
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    
    for step in range(max_steps):
        if render:
            frame = env.render()
            frames.append(frame)
        
        # Get action
        if agent is None:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict(obs, deterministic=True)
        
        # Step
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return frames, total_reward, step + 1


def train_ppo(env_name='CartPole-v1', timesteps=50000, save_path='ppo_model'):
    """Train a PPO agent."""
    print(f"\n{'='*60}")
    print("TRAINING PPO AGENT")
    print(f"{'='*60}\n")
    
    # Create environment
    env = gym.make(env_name)
    
    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0
    )
    
    print("✓ Agent created!")
    print(f"\nStarting training for {timesteps} timesteps...")
    
    # Train
    callback = ProgressCallback(check_freq=2000)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    
    # Save
    model.save(save_path)
    print(f"\n✓ Training complete! Model saved to '{save_path}.zip'")
    
    # Plot training progress
    plot_training_progress(callback.episode_rewards)
    
    env.close()
    return model, callback.episode_rewards


def plot_training_progress(rewards, window=50):
    """Plot training rewards."""
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(range(window-1, len(rewards)), moving_avg, 
             linewidth=2, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    final_rewards = rewards[-min(500, len(rewards)):]
    plt.hist(final_rewards, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(f'Reward Distribution (Last {len(final_rewards)} Episodes)')
    plt.axvline(np.mean(final_rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(final_rewards):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("✓ Training plot saved to 'training_progress.png'")
    plt.show()


def test_agent(env_name='CartPole-v1', model_path='ppo_model', n_episodes=10):
    """Test a trained agent."""
    print(f"\n{'='*60}")
    print("TESTING TRAINED AGENT")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    print(f"✓ Model loaded from '{model_path}.zip'")
    
    # Create environment
    env = gym.make(env_name, render_mode='rgb_array')
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Show one episode
    frames, reward, steps = render_episode(env, agent=model, seed=42)
    print(f"\nSample Episode:")
    print(f"  Reward: {reward:.1f}")
    print(f"  Steps: {steps}")
    
    env.close()
    return model


def compare_agents(env_name='CartPole-v1', model_path='ppo_model', n_episodes=10):
    """Compare random vs trained agent."""
    print(f"\n{'='*60}")
    print("COMPARING AGENTS")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    env = gym.make(env_name, render_mode='rgb_array')
    
    # Test both agents
    results = {}
    
    for name, agent in [('Random', None), ('PPO', model)]:
        rewards = []
        steps_list = []
        
        for ep in range(n_episodes):
            _, reward, steps = render_episode(env, agent, seed=ep, render=False)
            rewards.append(reward)
            steps_list.append(steps)
        
        results[name] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list)
        }
    
    # Display results
    print("\nResults:")
    for name, stats in results.items():
        print(f"\n{name} Agent:")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Mean Steps:  {stats['mean_steps']:.1f}")
    
    # Plot comparison
    names = list(results.keys())
    rewards = [results[name]['mean_reward'] for name in names]
    errors = [results[name]['std_reward'] for name in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, rewards, yerr=errors, capsize=10,
                   color=['#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title('Agent Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, max(rewards) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{reward:.1f}', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('agent_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved to 'agent_comparison.png'")
    plt.show()
    
    env.close()


def create_video(env_name='CartPole-v1', model_path='ppo_model', 
                output='ppo_demo.mp4', fps=30):
    """Create a video of the trained agent."""
    print(f"\n{'='*60}")
    print("CREATING VIDEO")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    env = gym.make(env_name, render_mode='rgb_array')
    
    # Render episode
    frames, reward, steps = render_episode(env, agent=model, seed=42)
    
    # Save video
    imageio.mimsave(output, frames, fps=fps)
    
    print(f"✓ Video saved to '{output}'")
    print(f"  Reward: {reward:.1f}")
    print(f"  Steps: {steps}")
    print(f"  Duration: {len(frames)/fps:.1f}s")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Gymnasium Quickstart with PPO'
    )
    parser.add_argument('--train', action='store_true',
                       help='Train a new PPO agent')
    parser.add_argument('--test', action='store_true',
                       help='Test a trained agent')
    parser.add_argument('--compare', action='store_true',
                       help='Compare random vs trained agent')
    parser.add_argument('--video', action='store_true',
                       help='Create video of trained agent')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name (default: CartPole-v1)')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps (default: 50000)')
    parser.add_argument('--model', type=str, default='ppo_model',
                       help='Model path (default: ppo_model)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.train, args.test, args.compare, args.video]):
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START EXAMPLES")
        print("="*60)
        print("\n1. Train an agent:")
        print("   python quickstart_ppo.py --train")
        print("\n2. Test the trained agent:")
        print("   python quickstart_ppo.py --test")
        print("\n3. Compare random vs trained:")
        print("   python quickstart_ppo.py --compare")
        print("\n4. Create a video:")
        print("   python quickstart_ppo.py --video")
        print("\n5. Train on different environment:")
        print("   python quickstart_ppo.py --train --env LunarLander-v3 --timesteps 100000")
        return
    
    # Execute requested actions
    if args.train:
        train_ppo(args.env, args.timesteps, args.model)
    
    if args.test:
        test_agent(args.env, args.model)
    
    if args.compare:
        compare_agents(args.env, args.model)
    
    if args.video:
        create_video(args.env, args.model)


if __name__ == '__main__':
    main()
