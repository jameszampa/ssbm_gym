import os
import re
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
import start_servers
time.sleep(5)
NUM_SERVERS = start_servers.TOTAL_NUM_PORTS

char1 = 'peach'
char2 = 'falcon'
ts = time.time()
models_dir = f"models/{char1}_{char2}_{int(ts)}/"
logdir = f"logs/{char1}_{char2}_{int(ts)}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# models_dir = "models/1675110384/"
# logdir = "logs/1675110384/"

env = make_vec_env(MeleeSelfPlay, n_envs=32 * 2 * NUM_SERVERS, env_kwargs={ 'model_name' : 'PPO', 'char1': char1, 'char2': char2, 'startingPort': start_servers.STARTING_PORT})
atexit.register(env.close)

# 
def get_latest_model(models_dir):
    max_iter = 0
    filename_iter = None
    for filename in os.listdir(models_dir):
        match = int(re.search(r"(\d\d+)", filename)[1])
        print(match)
        iteration = match
        if iteration > max_iter:
            max_iter = iteration
            filename_iter = filename
    return max_iter, filename_iter

#max_iter, filename = get_latest_model(models_dir)
#model_path = f"{models_dir}/{filename}"
#model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=logdir)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000000
#iters = max_iter / TIMESTEPS
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")