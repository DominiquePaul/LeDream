"""
Quick test to verify all dependencies are installed correctly.
Run this before starting the tutorials.
"""

import sys


def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    print("-" * 60)
    
    tests = []
    
    # Test gymnasium
    try:
        import gymnasium as gym
        print(f"✓ gymnasium {gym.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ gymnasium - {e}")
        tests.append(False)
    
    # Test stable-baselines3
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        print(f"✓ stable-baselines3 {stable_baselines3.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ stable-baselines3 - {e}")
        tests.append(False)
    
    # Test numpy
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ numpy - {e}")
        tests.append(False)
    
    # Test matplotlib
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ matplotlib - {e}")
        tests.append(False)
    
    # Test imageio
    try:
        import imageio
        print(f"✓ imageio {imageio.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ imageio - {e}")
        tests.append(False)
    
    # Test tqdm
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ tqdm - {e}")
        tests.append(False)
    
    print("-" * 60)
    
    if all(tests):
        print("\n🎉 All dependencies installed correctly!")
        return True
    else:
        print("\n⚠️  Some dependencies are missing.")
        print("Run: uv sync")
        return False


def test_environment():
    """Test creating a simple environment."""
    print("\n\nTesting environment creation...")
    print("-" * 60)
    
    try:
        import gymnasium as gym
        
        # Create environment
        env = gym.make('CartPole-v1')
        print("✓ Environment created: CartPole-v1")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful, reward: {reward}")
        
        # Close
        env.close()
        print("✓ Environment closed")
        
        print("-" * 60)
        print("🎉 Environment test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        print("-" * 60)
        return False


def test_ppo():
    """Test creating a PPO agent."""
    print("\n\nTesting PPO agent creation...")
    print("-" * 60)
    
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        
        # Create environment
        env = gym.make('CartPole-v1')
        
        # Create agent
        model = PPO('MlpPolicy', env, verbose=0)
        print("✓ PPO agent created")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        print(f"✓ Prediction successful, action: {action}")
        
        # Close
        env.close()
        
        print("-" * 60)
        print("🎉 PPO test passed!")
        return True
        
    except Exception as e:
        print(f"✗ PPO test failed: {e}")
        print("-" * 60)
        return False


def main():
    """Run all tests."""
    print("="*60)
    print(" " * 15 + "SETUP VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(test_imports())
    results.append(test_environment())
    results.append(test_ppo())
    
    # Summary
    print("\n" + "="*60)
    print(" " * 20 + "SUMMARY")
    print("="*60)
    
    if all(results):
        print("\n✅ All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Run quick demo: python demo_visual.py")
        print("  2. Or open notebook: jupyter notebook gymnasium_quickstart.ipynb")
        print("  3. Or train from CLI: python quickstart_ppo.py --train")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nTo install dependencies:")
        print("  uv sync")
        return 1


if __name__ == '__main__':
    sys.exit(main())
