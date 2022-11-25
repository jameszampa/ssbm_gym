

import os
import gym
from stable_baselines3 import PPO
from sb3_ssbm_env_render import CustomEnv

models_dir = "models/1667086758/"

options = dict(
    render=True,
    player1='ai',
    player2='cpu',
    char1='fox',
    char2='falco',
    cpu2=7,
    stage='battlefield',
)

env = CustomEnv(options=options)
env.reset()

max_iter = 0
for filename in os.listdir(models_dir):
    iteration = int(filename[:-4])
    if iteration > max_iter:
        max_iter = iteration
model_path = f"{models_dir}/{str(max_iter)}.zip"
model = PPO.load(model_path, env=env)


while True:
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
        #print(rewards)