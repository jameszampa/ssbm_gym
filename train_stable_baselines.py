import os
import re
import time
import random
import atexit
import platform
from ssbm_gym import EnvVec
from MeleeSelfPlay import MeleeSelfPlay
from stable_baselines3 import PPO, A2C
import numpy as np
from ssbm_gym.ssbm_env import SubprocVecEnv, make_env
from stable_baselines3.common.env_util import make_vec_env
from constants import *
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

parser = argparse.ArgumentParser()

parser.add_argument("char")
parser.add_argument("gpu", nargs='?', default=1)
parser.add_argument("resume", nargs='?', default=False)
parser.add_argument("model_dir", nargs='?', default=None)


args = parser.parse_args()

time.sleep(15)

NUM_SERVERS = TOTAL_NUM_PORTS

char = args.char
gpu_id = args.gpu
ts = time.time()
if not args.resume:
    models_dir = f"models/{char}_{int(ts)}/"
    logdir = f"logs/{char}_{int(ts)}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
else:
    models_dir = args.model_dir
    logdir = os.path.join('logs', args.model_dir.split(os.sep)[-1])

env = make_vec_env(MeleeSelfPlay, n_envs=26, env_kwargs={ 'model_name' : 'PPO', 'char': char, 'startingPort': STARTING_PORT, 'render': False, 'logdir': None})
env = VecCheckNan(env, raise_exception=True)
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

TIMESTEPS = 10000000
#iters = max_iter / TIMESTEPS


if not args.resume:
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda:' + str(gpu_id))
    iters = 0
else:
    max_iter, filename = get_latest_model(models_dir)
    model_path = f"{models_dir}/{filename}"
    model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=logdir, device='cuda:' + str(gpu_id))
    iters = int(max_iter / TIMESTEPS)

model.save(f"{models_dir}/{TIMESTEPS*iters}")
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")