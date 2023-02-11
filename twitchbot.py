import os
import re
import sys
import time
import json
import melee
import signal
import random
import requests
import threading

from twitchio.ext import commands
from twitchio.ext import routines
from ssbm_gym.gen_code import stage_ids
import subprocess
import psutil


with open('twitch_creds.json', 'r') as f:
    j_obj = json.loads(f.read())

BOT_ACCESS_TOKEN = j_obj['BOT_ACCESS_TOKEN']
BOT_CLIENT_ID = j_obj['BOT_CLIENT_ID']
BOT_PREFIX='!'
CHANNEL='ssbm_ai'
CHANNEL_ID='837561294'
BOT_CLIENT_SECRET = j_obj['BOT_CLIENT_SECRET']
CHANNEL_CLIENT_ID = j_obj['CHANNEL_CLIENT_ID']
CHANNEL_ACCESS_TOKEN = j_obj['CHANNEL_ACCESS_TOKEN']


# console = melee.Console(path="/home/james/Downloads/squashfs-root/usr/bin/",
#                         slippi_address="127.0.0.1",
#                         logger=None)

# controllerP1 = melee.Controller(console=console,
#                                 port=1,
#                                 type=melee.ControllerType.STANDARD)

# controllerP2 = melee.Controller(console=console,
#                                 port=2,
#                                 type=melee.ControllerType.STANDARD)


# def signal_handler(sig, frame):
#     console.stop()
#     sys.exit(0)


# signal.signal(signal.SIGINT, signal_handler)
# console.run(iso_path="./ISOs/SSBM.iso")


# success = console.connect()
# if not success:
#     raise ValueError("Failed to connect to libmelee console")

# controllerP1.connect()
# #controllerP2.connect()
# gamestate = console.step()
# while gamestate is None:
#     gamestate = console.step()

# while not gamestate.menu_state == melee.Menu.UNKNOWN_MENU:
#     # "step" to the next frame
#     gamestate = console.step()
#     if gamestate is None:
#         continue

# for i in range(100):
#     button = melee.enums.Button("START")
#     controllerP1.press_button(button)
#     gamestate = console.step()
#     button = melee.enums.Button("A")
#     controllerP1.press_button(button)
#     gamestate = console.step()
# time.sleep(2)

# for i in range(100):
#     button = melee.enums.Button("START")
#     controllerP1.press_button(button)
#     gamestate = console.step()
#     button = melee.enums.Button("A")
#     controllerP1.press_button(button)
#     gamestate = console.step()
#     time.sleep(0.1)
PORT = 20040
NUM_INSTANCES = 1
processes = []
p = subprocess.Popen(["./launch_server_main.sh", str(PORT)])
process = psutil.Process(p.pid)
process.nice(psutil.IOPRIO_CLASS_RT)
processes.append(process)
for i in range(1):
    p = subprocess.Popen(["./launch_server.sh", str(PORT + i + 1)])
    process = psutil.Process(p.pid)
    process.nice(psutil.IOPRIO_CLASS_RT)
    processes.append(process)
time.sleep(10)
setting_up = time.time()
char_model_dir = {
    'falco' : 'models/falco_1675869554',
    'falcon' : 'models/falco_1675869554',
    'fox' : 'models/falco_1675869554',
    'ics' : 'models/falco_1675869554',
    'marth' : 'models/falco_1675869554',
    'pikachu' : 'models/falco_1675869554',
    'samus' : 'models/falco_1675869554',
    'sheik' : 'models/falco_1675869554',
    'yoshi' : 'models/falco_1675869554',
}


def get_key_info():
    environment_ip = "127.0.0.1"
    port = PORT + 1
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"http://{environment_ip}:{port}/instance_info", headers=headers).json()
    return response['keys']


class displayInstance:
    def __init__(self, char1, char2, stage):
        self.setupP1 = subprocess.Popen([f"./display_latest.py {char1} {char_model_dir[char1]} {stage}"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        time.sleep(1)
        self.setupP2 = subprocess.Popen([f"./display_latest.py {char2} {char_model_dir[char2]} {stage}"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        self.ids = []
        time.sleep(10)


    def close(self):
        os.killpg(os.getpgid(self.setupP1.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.setupP2.pid), signal.SIGTERM)


class Bot(commands.Bot):
    def __init__(self):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        # prefix can be a callable, which returns a list of strings or a string...
        # initial_channels can also be a callable which returns a list of strings...
        super().__init__(token=BOT_ACCESS_TOKEN, prefix=BOT_PREFIX, initial_channels=[CHANNEL])
        self.instances = {}

        # get server ids for instances
        for i in range(NUM_INSTANCES):
            used_ids = []
            for j in range(NUM_INSTANCES):
                if j in self.instances.keys():
                    used_ids += self.instances[j].ids

            self.instances[i] = displayInstance("fox", "fox", "final_destination")
            all_ids = get_key_info()
            print(all_ids)
            for id in all_ids:
                if not id in used_ids:
                    self.instances[i].ids.append(id)
            print(self.instances[i].ids)


    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')
        self.hello.start('Test')

    
    # This routine will run every 5 seconds for 5 iterations.
    @routines.routine(minutes=10)
    async def hello(self, arg: str):
        pass
        #await self.connected_channels[0].send('test')


    @commands.command()
    async def set_matchup(self, ctx: commands.Context, *argv):
        global setting_up
        if time.time() - setting_up < 60:
            await self.connected_channels[0].send('Command executed too quickly, please wait')
            return
        if len(argv) < 4:
            await self.connected_channels[0].send('Too few arguments, try something like !set_matchup fox fox final_destination 0')
            return
        if not argv[0] in char_model_dir.keys():
            await self.connected_channels[0].send(f'Character: {argv[0]} is an unknown character or does not have a model. Check playable characters with !characters')
            return
        if not argv[1] in char_model_dir.keys():
            await self.connected_channels[0].send(f'Character: {argv[1]} is an unknown character or does not have a model. Check playable characters with !characters')
            return
        if not argv[2] in stage_ids.keys():
            await self.connected_channels[0].send(f'Stage: {argv[2]} is an unknown. Check playable stages with !stages')
            return
        if not argv[3] in ['0']:
            await self.connected_channels[0].send(f'Setup: {argv[3]} is an unknown. Try something like 0')
            return

        setting_up = time.time()
        await self.connected_channels[0].send(f'Setting matchup to {argv[0]} vs. {argv[1]} on {argv[2]} at setup {argv[3]}')
        environment_ip = "127.0.0.1"
        port = PORT + 1
        headers = {"Content-Type": "application/json"}
        json_message = {'gids': self.instances[int(argv[3])].ids}
        response = requests.post(f"http://{environment_ip}:{port}/close_instance", headers=headers, data=json.dumps(json_message))
        self.instances[int(argv[3])].close()
        del self.instances[int(argv[3])]

        used_ids = []
        for j in range(NUM_INSTANCES):
            if j in self.instances.keys():
                used_ids += self.instances[j].ids

        self.instances[int(argv[3])] = displayInstance(argv[0], argv[1], argv[2])
        all_ids = get_key_info()
        print(all_ids)
        for id in all_ids:
            if not id in used_ids:
                self.instances[i].ids.append(id)
        print(self.instances[i].ids)

    @commands.command()
    async def characters(self, ctx: commands.Context, *argv):
        o_str = "List of playable characters: "
        for char in char_model_dir.keys():
            o_str += char + " "
        await self.connected_channels[0].send(o_str)


    @commands.command()
    async def stages(self, ctx: commands.Context, *argv):
        o_str = "List of playable stages: "
        for stage in stage_ids.keys():
            o_str += stage + " "
        await self.connected_channels[0].send(o_str)



bot = Bot()
bot.run()