import os
import gym
import time
import json
import secrets
import datetime
import requests
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from ssbm_gym.embed import EmbedGame, numActions, numCharacters, numStages
from ssbm_gym.spaces import DiagonalActionSpace
from constants import STARTING_PORT



class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_rollout_end(self):
        moves_made = 0
        games_played = 0
        total_actions_without_move = 0
        avg_game_length = 0
        game_length_datapoints = 0
        for env in self.training_env.envs:
            moves_made += env.moves_played
            games_played += env.games_played
            #print(env.games_played)
            total_actions_without_move += env.actions_without_move
            if len(env.total_moves_history) > 0:
                for length in env.total_moves_history:
                    avg_game_length += length
                    game_length_datapoints += 1

        self.logger.record("custom/moves_made", moves_made)
        self.logger.record("custom/games_played", games_played)
        self.logger.record("custom/avg_actions_without_move", total_actions_without_move / len(self.training_env.envs))
        if game_length_datapoints > 0:
            self.logger.record("custom/avg_game_length", avg_game_length / game_length_datapoints)
        else:
            self.logger.record("custom/avg_game_length", -1)

    def _on_step(self):
        return True


class MeleeSelfPlay(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, model_name, render=False, startingPort=STARTING_PORT, frameLimit=10000, char='falcon', stage='random'):
        super(MeleeSelfPlay, self).__init__()
        spaces = {
            'player0_character': gym.spaces.Box(low=-1, high=1, shape=(numCharacters,)),
            'player0_action_state': gym.spaces.Box(low=-1, high=1, shape=(numActions,)),
            'player0_state': gym.spaces.Box(low=-1, high=1, shape=(10,)),
            'player1_character': gym.spaces.Box(low=-1, high=1, shape=(numCharacters,)),
            'player1_action_state': gym.spaces.Box(low=-1, high=1, shape=(numActions,)),
            'player1_state': gym.spaces.Box(low=-1, high=1, shape=(10,)),
            'stage': gym.spaces.Box(low=-1, high=1, shape=(numStages,)),
        }
        self.observation_space = gym.spaces.Dict(spaces)
        self.metadata = {}
        self.doesRender = render
        self.frame_limit = frameLimit
        self.char = char
        self.stage = stage

        self.environment_ip = "127.0.0.1"
        self.port = startingPort
        headers = {"Content-Type": "application/json"}
        json_message = {'model_name':model_name, 'doesRender':self.doesRender, 'startingPort': startingPort, 'char': self.char, 'stage': self.stage}
        response = requests.post(f"http://{self.environment_ip}:{self.port}/assign_id", headers=headers, data=json.dumps(json_message))
        if 'error' in response.json().keys():
            raise ValueError(response.json()['error'])
        self.gid = response.json()['gid']
        self.port = response.json()['port']
        json_message = {'model_name':model_name, 'doesRender':self.doesRender, 'startingPort': startingPort, 'char': self.char, 'gid': self.gid, 'port': self.port, 'stage': self.stage}
        if self.port != startingPort:
            requests.post(f"http://{self.environment_ip}:{self.port}/assign_id", headers=headers, data=json.dumps(json_message))
            if 'error' in response.json().keys():
                raise ValueError(response.json()['error'])
        
        self.custom_action = DiagonalActionSpace()
        self.action_space = gym.spaces.Discrete(self.custom_action.n)


    def step(self, action):
        headers = {"Content-Type": "application/json"}
        json_message = {'action': int(action),
                        'gid': self.gid,
                        'frame_limit': self.frame_limit
        }
        response = requests.post(f"http://{self.environment_ip}:{self.port}/step", headers=headers, data=json.dumps(json_message)).json()
        if 'error' in response.keys():
            raise ValueError(response.json()['error'])

        self.observation = response['observation']
        self.done = response['done']
        self.reward = response['reward']
        info = {}
        return self.observation, self.reward, self.done, info


    def reset(self):
        self.done = False
        headers = {"Content-Type": "application/json"}
        json_message = {'gid': self.gid}
        response = requests.post(f"http://{self.environment_ip}:{self.port}/reset", headers=headers, data=json.dumps(json_message)).json()
        #print(response)
        if 'error' in response.keys():
            raise ValueError(response['error'])
        self.observation = response['observation']
        return self.observation


    def render(self, mode='human'):
        pass


    def close (self):
        headers = {"Content-Type": "application/json"}
        json_message = {'gid': self.gid}
        response = requests.post(f"http://{self.environment_ip}:{self.port}/close", headers=headers, data=json.dumps(json_message)).json()
        if 'error' in response.keys():
            raise ValueError(response['error'])