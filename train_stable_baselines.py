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


# Set up logging and model checkpoint directories
ts = time.time()
models_dir = f"models/{int(ts)}/"
logdir = f"logs/{int(ts)}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


env = make_vec_env(MeleeSelfPlay, n_envs=32 * 2, env_kwargs={ 'model_name' : 'PPO' })
atexit.register(env.close)

# Load stable_baselines3 PPO model

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 1000000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")