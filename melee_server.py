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


STARTING_PORT = 10000
MAX_PER_SERVER = 32


class MeleeInstance():
    def __init__(self, gid, model_name, doesRender):
        self.gid = gid
        self.hasOpponent = False
        self.opponentId = None
        self.model_name = model_name
        self.pid = None
        self.doesRender = doesRender


class DolphinInstance():
    def __init__(self, doesRender):
        self.options = dict(
            render=doesRender,
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


instances = {}
dolphins = {}
_embed_obs = EmbedGame(flat=False)
custom_action = DiagonalActionSpace()
curr_server_load_idx = 0

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
                dolphins[gid + '_' + request_json['gid']] = DolphinInstance(instances[gid].doesRender or instances[request_json['gid']].doesRender)
                instances[gid].opponentId = request_json['gid']
                instances[request_json['gid']].opponentId = gid
                instances[gid].pid = 0
                instances[request_json['gid']].pid = 1
                print(f"Found Game! {gid} vs. {instances[gid].opponentId}")
                foundGame = True
                break
        if not foundGame and not instances[request_json['gid']].hasOpponent:
            instances[request_json['gid']].waitingForOpponent = True
            instances[request_json['gid']].hasOpponent = False
            print(f"Waiting for game... {request_json['gid']}")
            obs = {
                'player0_character': list(np.zeros(shape=(numCharacters,))),
                'player0_action_state': list(np.zeros(shape=(numActions,))),
                'player0_state': list(np.zeros(shape=(10,))),
                'player1_character': list(np.zeros(shape=(numCharacters,))),
                'player1_action_state': list(np.zeros(shape=(numActions,))),
                'player1_state': list(np.zeros(shape=(10,))),
                'stage': list(np.zeros(shape=(numStages,))),
            }
            return json.dumps({'observation': obs}), 200
        else:
            print(f"Resetting dolphin...")
            try:
                obs = dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].dolphin.reset()
                dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].prev_obs = obs
            except:
                obs = dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].dolphin.reset()
                dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].prev_obs = obs
            print(f"Done!")
            return json.dumps({'observation': embed_obs(obs)}), 200
    else:
        return Response(status=400)


@app.route('/assign_id', methods=["POST"])
def assign_id():
    global curr_server_load_idx
    if request.method == 'POST':
        request_json = request.get_json(force=True)
        gid = secrets.token_urlsafe(16)
        used_gids = [instances[inst_key].gid for inst_key in instances.keys()]
        while gid in used_gids:
            gid = secrets.token_urlsafe(16)
        instances[gid] = MeleeInstance(gid, request_json["model_name"], request_json["doesRender"])
        
        if len(instances) <= MAX_PER_SERVER * 2 * (curr_server_load_idx + 1):
            port = int(request_json["startingPort"]) + curr_server_load_idx
        else:
            curr_server_load_idx += 1
            port = int(request_json["startingPort"]) + curr_server_load_idx

        return json.dumps({'gid': gid, 'port': port}), 200
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

    if instances[request_json['gid']].pid == 0:
        dolphins[boardKey].p1_action = custom_action.from_index(int(request_json['action']))
        # print(f"Updating action for P1 {int(request_json['action'])}")
    else:
        dolphins[boardKey].p2_action = custom_action.from_index(int(request_json['action']))
        # print(f"Updating action for P2 {int(request_json['action'])}")

    obs = dolphins[boardKey].prev_obs
    done = False
    if not dolphins[boardKey].p1_action is None and not dolphins[boardKey].p2_action is None:
        obs = dolphins[boardKey].dolphin.step([dolphins[boardKey].p1_action, dolphins[boardKey].p2_action])
        dolphins[boardKey].p1_action = None
        dolphins[boardKey].p2_action = None
        done = is_terminal(obs, int(request_json['frame_limit']))
        # print(f"Taking dolphin step")

    if not dolphins[boardKey].prev_obs is None:
        reward = compute_reward(dolphins[boardKey].prev_obs, obs, instances[request_json['gid']].pid)
    else:
        reward = 0

    dolphins[boardKey].prev_obs = deepcopy(obs)
    json_message = {
        'observation': embed_obs(obs),
        'done': done,
        'reward': reward
    }
    return json.dumps(json_message), 200