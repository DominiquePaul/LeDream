import gymnasium as gym

env = gym.make('LunarLander-v3', render_mode='human')
obs, info = env.reset()

for step in range(150):
    env.render()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break

env.close()