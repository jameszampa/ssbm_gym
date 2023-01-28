import os
import time
import random
import atexit
import platform
from ssbm_gym import EnvVec
from MeleeSelfPlay import MeleeSelfPlay
from stable_baselines3 import PPO
import numpy as np
from ssbm_gym.ssbm_env import SubprocVecEnv, make_env
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
model = PPO("MultiInputPolicy", env, tensorboard_log=logdir)

TIMESTEPS = 1000000
iters = int(max_iter / TIMESTEPS)
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")