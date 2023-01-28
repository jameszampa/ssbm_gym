import os
import re
import time
import json
import random
import secrets
from copy import deepcopy
import numpy as np
from flask import Flask, request, Response
from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm_env import CHARACTERS, COMPETITVE_STAGES
from ssbm_gym.embed import EmbedGame, numActions, numCharacters, numStages
from ssbm_gym.spaces import DiagonalActionSpace


app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)


class MeleeInstance():
    def __init__(self, gid, model_name):
        self.gid = gid
        self.hasOpponent = False
        self.opponentId = None
        self.model_name = model_name
        self.pid = None


class DolphinInstance():
    def __init__(self):
        self.options = dict(
            render=False,
            player1='ai',
            player2='ai',
            char1='falcon',
            char2='falcon',
            stage=COMPETITVE_STAGES[random.randint(0, len(COMPETITVE_STAGES) - 1)],
        )
        self.dolphin = DolphinAPI(**self.options)
        self.p1_action = None
        self.p2_action = None
        self.prev_obs = None
        self.custom_action = DiagonalActionSpace()

    
    def step(self, action_pid):
        action = action_pid[0]
        pid = action_pid[1]
        if not self.p1_action is None and not self.p2_action is None:
            obs = self.dolphin.step([self.custom_action.from_index(self.p1_action)])
            self.p1_action = None
            self.p2_action = None
        else:
            if pid == 0:
                self.p1_action = action
            else:
                self.p2_action = action
            obs = self.prev_obs
        reward = compute_reward(self.prev_obs, obs, pid)
        done = is_terminal(obs, 100000)
        infos = {}
        self.prev_obs = deepcopy(obs)
        return embed_obs(obs), reward, done, infos


    def reset(self):
        obs = self.dolphin.reset()
        self.prev_obs = deepcopy(obs)
        return embed_obs(obs)


    def close(self):
        self.dolphin.close()


instances = {}
dolphins = {}
_embed_obs = EmbedGame(flat=False)



def embed_obs(obs):
    obs = _embed_obs(obs)
    stable_safe_obs = {
        'player0_character': obs['player0']['character'],
        'player0_action_state': obs['player0']['action_state'],
        'player0_state': obs['player0']['state'],
        'player1_character': obs['player1']['character'],
        'player1_action_state': obs['player1']['action_state'],
        'player1_state': obs['player1']['state'],
        'stage': obs['stage'],
    }
    return stable_safe_obs


def isDying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player.action_state <= 0xA


def is_terminal(obs, frame_limit):
    return obs.frame >= frame_limit


def compute_reward(prev_obs, obs, pid):
    r = 0.0
    if prev_obs is not None:
        # This is necesarry because the character might be dying during multiple frames
        if not isDying(prev_obs.players[pid]) and \
            isDying(obs.players[pid]):
            r -= 1.0
            
        # We give a reward of -0.01 for every percent taken. The max() ensures that not reward is given when a character dies
        r -= 0.01 * max(0, obs.players[pid].percent - prev_obs.players[pid].percent)

        # Here we reverse the pid list to access the opponent state
        if not isDying(prev_obs.players[1-pid]) and \
            isDying(obs.players[1-pid]):
            r += 1.0

        r += 0.01 * max(0, obs.players[pid].percent - prev_obs.players[pid].percent) 

        return r


### Vectorizing


import multiprocessing
import cloudpickle
import pickle


def make_env(env_class):
    def _init():
        env = env_class()
        return env
    return _init


def EnvVec(env_class, num_envs):
    return SubprocVecEnv([make_env(env_class=env_class) for _ in range(num_envs)])


class CloudpickleWrapper(object):
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(): 
    def __init__(self, env_fns, start_method=None):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        self.metadata = {}
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'fork' in multiprocessing.get_all_start_methods()
            start_method = 'fork' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # self.remotes[0].send(('get_spaces', None))
        # self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions, pid):
        obs, rew, done, info = self.step_async(actions, pid)
        return obs, rew, done, info

    def step_async(self, actions, pid):
        #for remote, action in zip(self.remotes, actions):
        self.remotes[0].send(('step', [actions, pid]))
        obs, rew, done, info = self.remotes[0].recv()
        return obs, rew, done, info
        #self.waiting = True

    def set_waiting(self, waiting):
        self.waiting = waiting

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True


    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]


    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()


    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]


    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


@app.route('/reset', methods=["POST"])
def reset_game():
    if request.method == 'POST':
        #try:
        
        foundGame = False
        request_json = request.get_json(force=True)
        print(f"Reset! {request_json['gid']}")
        for gid in instances.keys():
            if gid != request_json['gid'] and not instances[gid].hasOpponent:
                instances[gid].hasOpponent = True
                instances[request_json['gid']].hasOpponent = True
                instances[gid].waitingForOpponent = False
                instances[request_json['gid']].waitingForOpponent = False
                try:
                    try:
                        dolphins[gid + '_' + request_json['gid']].dolphin.close()
                        del dolphins[gid + '_' + request_json['gid']]
                    except:
                        dolphins[request_json['gid'] + '_' + gid].dolphin.close()
                        del dolphins[request_json['gid'] + '_' + gid]
                except:
                    pass
                dolphins[gid + '_' + request_json['gid']] = EnvVec(DolphinInstance, 1)
                instances[gid].opponentId = request_json['gid']
                instances[request_json['gid']].opponentId = gid
                instances[gid].pid = 0
                instances[request_json['gid']].pid = 1
                print(f"Found Game! {gid} vs. {instances[gid].opponentId}")
                foundGame = True
                break
        if not foundGame and not instances[request_json['gid']].hasOpponent:
            instances[request_json['gid']].waitingForOpponent = True
            print(f"Waiting for game... {request_json['gid']}")
            obs = {
                'player0_character': np.zeros(shape=(numCharacters,)),
                'player0_action_state': np.zeros(shape=(numActions,)),
                'player0_state': np.zeros(shape=(10,)),
                'player1_character': np.zeros(shape=(numCharacters,)),
                'player1_action_state': np.zeros(shape=(numActions,)),
                'player1_state': np.zeros(shape=(10,)),
                'stage': np.zeros(shape=(numStages,)),
            }
            return json.dumps({'observation': obs}), 200
        else:
            print(f"Resetting dolphin...")
            try:
                obs = dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].reset()
            except:
                obs = dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].reset()
            print(f"Done!")
            return json.dumps({'observation': obs[0]}), 200
    else:
        return Response(status=400)


@app.route('/assign_id', methods=["POST"])
def assign_id():
    if request.method == 'POST':
        request_json = request.get_json(force=True)
        gid = secrets.token_urlsafe(16)
        used_gids = [instances[inst_key].gid for inst_key in instances.keys()]
        while gid in used_gids:
            gid = secrets.token_urlsafe(16)
        instances[gid] = MeleeInstance(gid, request_json["model_name"])
        return json.dumps({'gid': gid}), 200
    else:
        return Response(status=400)


@app.route('/step', methods=["POST"])
def step():
    if not request.method == 'POST':
        return Response(status=400)

    request_json = request.get_json(force=True)
    opponentId = instances[request_json['gid']].opponentId
    try:
        _ = dolphins[request_json['gid'] + '_' + opponentId]
        boardKey = request_json['gid'] + '_' + opponentId
    except:
        _ = dolphins[opponentId + '_' + request_json['gid']]
        boardKey = opponentId + '_' + request_json['gid']

    action = request_json['action']
    obs, reward, done, infos = dolphins[boardKey].step(action, instances[request_json['gid']].pid)
    json_message = {
        'observation': obs,
        'done': done,
        'reward': reward
    }
    return json.dumps(json_message), 200