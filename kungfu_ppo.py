import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("ALE/KungFuMaster-v5", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=5000000)
model.save("ppo_kungfu")

del model

model = PPO.load("ppo_kungfu")

obs = vec_env.reset()
while True:
    action, states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")