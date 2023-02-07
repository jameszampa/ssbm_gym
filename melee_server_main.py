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
from constants import *
from MeleeInstance import MeleeInstance
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)


MAX_PER_SERVER = 32
instances = {}
curr_server_load_idx = 0
port_count = {}
for i in range(TOTAL_NUM_PORTS):
    port_count[i] = {}
    for char in CHARACTERS:
        port_count[i][char] = 0


@app.route('/assign_id', methods=["POST"])
def assign_id():
    global curr_server_load_idx
    if request.method == 'POST':
        request_json = request.get_json(force=True)
        gid = secrets.token_urlsafe(16)
        used_gids = [instances[inst_key].gid for inst_key in instances.keys()]
        while gid in used_gids:
            gid = secrets.token_urlsafe(16)
        
        min_port = None
        for i in range(TOTAL_NUM_PORTS):
            if min_port is None:
                min_port = i
            elif port_count[i][request_json["char"]] < port_count[min_port][request_json["char"]]:
                min_port = i
            

        port = int(request_json["startingPort"]) + min_port + 1
        port_count[min_port][request_json["char"]] += 1

        instances[gid] = MeleeInstance(gid, request_json["model_name"], request_json["doesRender"], request_json["char"], port)

        return json.dumps({'gid': gid, 'port': port}), 200
    else:
        return Response(status=400)