

import os
import re
import gym
import time
import random
import atexit
import platform
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ssbm_gym import EnvVec
from ssbm_gym.ssbm_env import SubprocVecEnv, make_env
from MeleeSelfPlay import MeleeSelfPlay

import subprocess

TOTAL_NUM_PORTS = 1

processes = []
for i in range(TOTAL_NUM_PORTS):
    p = subprocess.Popen(["./launch_server.sh", str(20100 + i)])
    processes.append(p)

time.sleep(5)

ts = time.time()
# models_dir = f"models/{int(ts)}/"
# logdir = f"logs/{int(ts)}/"

# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)

# if not os.path.exists(logdir):
# 	os.makedirs(logdir)

models_dir = "models/marth_fox_1675534734/"

char1 = 'marth'
char2 = 'fox'

# Load stable_baselines3 PPO model

# model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)
def get_latest_model(models_dir):
    max_iter = 0
    filename_iter = None
    for filename in os.listdir(models_dir):
        match = int(re.search(r"(\d\d+)", filename)[1])
        iteration = match
        if iteration > max_iter:
            max_iter = iteration
            filename_iter = filename
    return max_iter, filename_iter

max_iter, filename = get_latest_model(models_dir)
model_path = f"{models_dir}/{filename}"
env = make_vec_env(MeleeSelfPlay, n_envs=2, env_kwargs={ 'model_name' : 'PPO' , 'render' : True, 'startingPort': 20000, 'frameLimit': 1000000, 'char1': char1, 'char2': char2})
atexit.register(env.close)
model = PPO.load(model_path, env=env, verbose=1)
obs = env.reset()
done = False

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
    isDone = False
    for d in dones:
        if d:
            isDone = True
    
    if not isDone:
        continue
    new_max_iter, filename = get_latest_model(models_dir)
    if new_max_iter > max_iter:
        max_iter, filename = get_latest_model(models_dir)
        model_path = f"{models_dir}/{filename}"
        model = PPO.load(model_path, env=env, verbose=1)
        obs = env.reset()
    else:
        obs = env.reset()
