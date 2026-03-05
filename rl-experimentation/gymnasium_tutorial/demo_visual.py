"""
Quick Visual Demo - Train and visualize a PPO agent in ~2 minutes

This script trains a PPO agent on CartPole and immediately shows
the before/after performance with side-by-side visualization.

Perfect for: Quick demonstrations, testing setup, learning basics
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import warnings
warnings.filterwarnings('ignore')


def quick_render(env, agent=None, seed=42):
    """Quickly render one episode."""
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0
    
    for _ in range(500):
        frame = env.render()
        frames.append(frame)
        
        if agent is None:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return frames, total_reward


def main():
    print("=" * 70)
    print(" " * 15 + "GYMNASIUM + PPO VISUAL DEMO")
    print("=" * 70)
    
    # Setup
    print("\n[1/5] Setting up environment...")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # Show random performance
    print("[2/5] Testing random agent (baseline)...")
    random_frames, random_reward = quick_render(env, agent=None)
    print(f"      → Random agent reward: {random_reward:.1f}")
    
    # Train PPO
    print("[3/5] Training PPO agent (this takes ~60 seconds)...")
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=30000, progress_bar=True)
    print("      ✓ Training complete!")
    
    # Test trained agent
    print("[4/5] Testing trained agent...")
    trained_frames, trained_reward = quick_render(env, agent=model)
    print(f"      → Trained agent reward: {trained_reward:.1f}")
    
    # Evaluate properly
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"      → Mean reward over 10 episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Visualize side by side
    print("[5/5] Creating visualization...")
    
    # Create side-by-side video
    max_frames = max(len(random_frames), len(trained_frames))
    
    # Pad shorter video
    if len(random_frames) < max_frames:
        random_frames.extend([random_frames[-1]] * (max_frames - len(random_frames)))
    if len(trained_frames) < max_frames:
        trained_frames.extend([trained_frames[-1]] * (max_frames - len(trained_frames)))
    
    # Create side-by-side frames
    comparison_frames = []
    for rf, tf in zip(random_frames, trained_frames):
        # Add labels
        combined = np.hstack([rf, tf])
        comparison_frames.append(combined)
    
    # Save videos
    imageio.mimsave('random_agent.mp4', random_frames[:len(random_frames)], fps=30)
    imageio.mimsave('trained_agent.mp4', trained_frames[:len(trained_frames)], fps=30)
    imageio.mimsave('comparison.mp4', comparison_frames, fps=30)
    
    print("\n✓ Videos saved:")
    print("  • random_agent.mp4")
    print("  • trained_agent.mp4")
    print("  • comparison.mp4 (side-by-side)")
    
    # Create performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    agents = ['Random', 'Trained PPO']
    rewards = [random_reward, trained_reward]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(agents, rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 500)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{reward:.0f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Show frame samples
    ax2.axis('off')
    ax2.set_title('Visual Comparison', fontsize=14, fontweight='bold')
    
    # Get middle frames
    sample_idx = len(random_frames) // 2
    comparison_sample = comparison_frames[sample_idx]
    ax2.imshow(comparison_sample)
    
    # Add text labels
    h, w, _ = random_frames[0].shape
    ax2.text(w//2, -20, 'RANDOM', ha='center', va='top', 
            fontsize=12, fontweight='bold', color='#ff7f0e')
    ax2.text(w + w//2, -20, 'TRAINED', ha='center', va='top',
            fontsize=12, fontweight='bold', color='#2ca02c')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("  • performance_comparison.png")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE! 🎉")
    print("=" * 70)
    print("\nKey Takeaways:")
    print(f"  • Random agent achieved: {random_reward:.1f} reward")
    print(f"  • Trained agent achieved: {trained_reward:.1f} reward")
    print(f"  • Improvement: {trained_reward - random_reward:.1f} points")
    print(f"  • Training time: ~60 seconds")
    print("\nNext Steps:")
    print("  1. Check out gymnasium_quickstart.ipynb for detailed tutorial")
    print("  2. Run: python quickstart_ppo.py --train --timesteps 100000")
    print("  3. Try different environments like LunarLander-v3")
    print("\n")
    
    env.close()
    plt.show()


if __name__ == '__main__':
    main()
