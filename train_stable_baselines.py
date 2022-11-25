import os
import time
import random
import atexit
import platform
from ssbm_gym import EnvVec
from sb3_ssbm_env import CustomEnv
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

# Create SSBM Env
options = dict(
    render=False,
    player1='ai',
    player2='cpu',
    char1='fox',
    char2='falco',
    cpu2=7,
    stage='battlefield',
)

if platform.system() == 'Windows':
    options['windows'] = True


#env = SubprocVecEnv([make_env(CustomEnv, 1e12, options) for i in range(8)])
#env = EnvVec(CustomEnv, 8, options=options)
env = make_vec_env(CustomEnv, n_envs=225, vec_env_cls=SubprocVecEnv)
obs = env.reset()
atexit.register(env.close)

# Load stable_baselines3 PPO model

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda:1')

TIMESTEPS = 1000000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")