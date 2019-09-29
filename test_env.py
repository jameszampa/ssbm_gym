import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('../')

from ssbm_env import SSBMEnv
import time
import argparse

parser = argparse.ArgumentParser()
SSBMEnv.update_parser(parser)

args = parser.parse_args()
print(args.__dict__)
print(args.iso)

env = SSBMEnv(**args.__dict__)
#cpu1=9, cpu2=9
env.reset()


for i in range(1000):
    env.step(None)

env.close()

