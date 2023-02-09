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
from MeleeInstance import MeleeInstance


app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)


MAX_PER_SERVER = 32


class DolphinInstance():
    def __init__(self, doesRender, char1, char2, stage):
        if not stage == 'random':
            self.stage = stage
        else:
            self.stage = COMPETITVE_STAGES[random.randint(0, len(COMPETITVE_STAGES) - 1)]

        self.options = dict(
            render=doesRender,
            player1='ai',
            player2='ai',
            char1=char1,
            char2=char2,
            stage=self.stage,
        )
        self.dolphin = DolphinAPI(**self.options)
        self.p1_action = None
        self.p2_action = None
        self.prev_obs1 = None
        self.prev_obs2 = None
        self.isReset = False
        self.stepRunning = False


instances = {}
dolphins = {}
_embed_obs = EmbedGame(flat=False)
custom_action = DiagonalActionSpace()


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

        r += 0.01 * max(0, obs.players[1-pid].percent - prev_obs.players[1-pid].percent) 

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
                dolphins[gid + '_' + request_json['gid']] = DolphinInstance(instances[gid].doesRender or instances[request_json['gid']].doesRender,
                                                                            instances[gid].char,
                                                                            instances[request_json['gid']].char,
                                                                            instances[gid].stage)
                #dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].prev_obs = None
                instances[gid].opponentId = request_json['gid']
                instances[request_json['gid']].opponentId = gid
                instances[gid].pid = 0
                instances[request_json['gid']].pid = 1
                print(f"Found Game! {instances[gid].char} vs. {instances[request_json['gid']].char}")
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
            
            try:
                if not dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].isReset:
                    print(f"Resetting dolphin...")
                    dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].isReset = True
                    obs = dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].dolphin.reset()
                    dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].prev_obs1 = obs
                    dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].isReset = False
                    dolphins[instances[request_json['gid']].opponentId + '_' + request_json['gid']].waitingToReset = False
                else:
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
            except:
                if not dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].isReset:
                    print(f"Resetting dolphin...")
                    dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].isReset = True
                    obs = dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].dolphin.reset()
                    dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].prev_obs1 = obs
                    dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].isReset = False
                    dolphins[request_json['gid'] + '_' + instances[request_json['gid']].opponentId].waitingToReset = False
                else:
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
            print(f"Done!")
            instances[instances[request_json['gid']].opponentId].isDone = False
            instances[request_json['gid']].isDone = False
            return json.dumps({'observation': embed_obs(obs)}), 200
    else:
        return Response(status=400)


@app.route('/assign_id', methods=["POST"])
def assign_id():
    if request.method == 'POST':
        request_json = request.get_json(force=True)
        gid = request_json['gid']
        instances[gid] = MeleeInstance(gid, request_json["model_name"], request_json["doesRender"], request_json["char"], request_json["port"], request_json["stage"])

        return Response(status=200)
    else:
        return Response(status=400)


@app.route('/step', methods=["POST"])
def step():
    if not request.method == 'POST':
        return Response(status=400)

    request_json = request.get_json(force=True)
    if not request_json['gid'] in instances.keys():
        obs = {
            'player0_character': list(np.zeros(shape=(numCharacters,))),
            'player0_action_state': list(np.zeros(shape=(numActions,))),
            'player0_state': list(np.zeros(shape=(10,))),
            'player1_character': list(np.zeros(shape=(numCharacters,))),
            'player1_action_state': list(np.zeros(shape=(numActions,))),
            'player1_state': list(np.zeros(shape=(10,))),
            'stage': list(np.zeros(shape=(numStages,))),
        }
        json_message = {
            'observation': obs,
            'done': False,
            'reward': 0
        }
        return json.dumps(json_message), 200


    opponentId = instances[request_json['gid']].opponentId
    boardKey = None
    try:
        _ = dolphins[request_json['gid'] + '_' + opponentId]
        boardKey = request_json['gid'] + '_' + opponentId
    except:
        _ = dolphins[opponentId + '_' + request_json['gid']]
        boardKey = opponentId + '_' + request_json['gid']

    if opponentId is None or boardKey is None or dolphins[boardKey].prev_obs1 is None:
        obs = {
            'player0_character': list(np.zeros(shape=(numCharacters,))),
            'player0_action_state': list(np.zeros(shape=(numActions,))),
            'player0_state': list(np.zeros(shape=(10,))),
            'player1_character': list(np.zeros(shape=(numCharacters,))),
            'player1_action_state': list(np.zeros(shape=(numActions,))),
            'player1_state': list(np.zeros(shape=(10,))),
            'stage': list(np.zeros(shape=(numStages,))),
        }
        json_message = {
            'observation': obs,
            'done': False,
            'reward': 0
        }
        return json.dumps(json_message), 200

    if instances[request_json['gid']].pid == 0:
        dolphins[boardKey].p1_action = custom_action.from_index(int(request_json['action']))
        #print(f"Updating action for P1 {int(request_json['action'])}")
    else:
        dolphins[boardKey].p2_action = custom_action.from_index(int(request_json['action']))
        #print(f"Updating action for P2 {int(request_json['action'])}")

    isDone = False
    if not dolphins[boardKey].p1_action is None and not dolphins[boardKey].p2_action is None and not dolphins[boardKey].stepRunning and instances[request_json['gid']].pid == 0:
        #print("gameStep!")
        dolphins[boardKey].stepRunning = True
        obs = dolphins[boardKey].dolphin.step([dolphins[boardKey].p1_action, dolphins[boardKey].p2_action])
        dolphins[boardKey].p1_action = None
        dolphins[boardKey].p2_action = None
        isDone = is_terminal(obs, int(request_json['frame_limit']))
        instances[request_json['gid']].isDone = isDone
        instances[opponentId].isDone = isDone
        #print("gameStep! Done")
        dolphins[boardKey].stepRunning = False
        dolphins[boardKey].prev_obs2 = deepcopy(dolphins[boardKey].prev_obs1)
        dolphins[boardKey].prev_obs1 = deepcopy(obs)
        # print(f"Taking dolphin step")
    done = False
    if (instances[request_json['gid']].isDone or instances[opponentId].isDone) and not dolphins[boardKey].waitingToReset:
        done = True
        dolphins[boardKey].waitingToReset = True

    if not dolphins[boardKey].prev_obs2 is None and not dolphins[boardKey].prev_obs1 is None:
        reward = compute_reward(dolphins[boardKey].prev_obs2, dolphins[boardKey].prev_obs1, instances[request_json['gid']].pid)
    else:
        reward = 0

    obs = embed_obs(dolphins[boardKey].prev_obs1)
    json_message = {
        'observation': obs,
        'done': done,
        'reward': reward
    }
    return json.dumps(json_message), 200


@app.route('/close', methods=["POST"])
def close():
    request_json = request.get_json(force=True)
    # opponentId = instances[request_json['gid']].opponentId
    # try:
    #     dolphins[opponentId + '_' + request_json['gid']].dolphin.close()
    #     del dolphins[opponentId + '_' + request_json['gid']]
    # except:
    #     dolphins[request_json['gid'] + '_' + opponentId].dolphin.close()
    #     del dolphins[request_json['gid'] + '_' + opponentId]
    # del instances[request_json['gid']]
    return Response(status=200)


@app.route('/close_instance', methods=["POST"])
def close_instance():
    try:
        for gid in instances.keys():
                opponentId = instances[gid].opponentId
                try:
                    dolphins[opponentId + '_' + gid].dolphin.close()
                    del dolphins[opponentId + '_' + gid]
                except:
                    dolphins[gid + '_' + opponentId].dolphin.close()
                    del dolphins[gid + '_' + opponentId]
                del instances[gid]
                del instances[opponentId]
    except:
        pass
    return Response(status=200)