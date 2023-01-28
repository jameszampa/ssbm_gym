

import os
import gym
from stable_baselines3 import PPO
from MeleeSelfPlay import MeleeSelfPlay
import atexit
from stable_baselines3.common.env_util import make_vec_env


models_dir = "models/1674786159/"
logdir = "logs/1674786159/"

env = make_vec_env(MeleeSelfPlay, n_envs=2, env_kwargs={ 'model_name' : 'PPO' })
atexit.register(env.close)

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
        obs, rewards, dones, info = env.step(action)
        done = False
        for isDone in dones:
            if isDone:
                done = True
        #env.render()
        #print(rewards)