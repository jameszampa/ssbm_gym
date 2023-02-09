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

processes = []
p = subprocess.Popen(["./launch_server_main.sh", str(20000)])
for i in range(1):
    p = subprocess.Popen(["./launch_server.sh", str(20000 + i + 1)])
    processes.append(p)
setting_up = time.time()

class Bot(commands.Bot):
    def __init__(self):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        # prefix can be a callable, which returns a list of strings or a string...
        # initial_channels can also be a callable which returns a list of strings...
        super().__init__(token=BOT_ACCESS_TOKEN, prefix=BOT_PREFIX, initial_channels=[CHANNEL])
        self.setupP1 = subprocess.Popen(["./display_latest.py fox models/fox_1675869554 final_destination"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        self.setupP2 = subprocess.Popen(["./display_latest.py fox models/fox_1675869554 final_destination"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        self.char_model_dir = {
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
        self.stage_list = stage_ids.keys()
        

        

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
        if len(argv) < 3:
            await self.connected_channels[0].send('Too few arguments, try something like !set_matchup fox fox final_destination')
            return
        if not argv[0] in self.char_model_dir.keys():
            await self.connected_channels[0].send(f'Character: {argv[0]} is an unknown character or does not have a model. Check playable characters with !characters')
            return
        if not argv[1] in self.char_model_dir.keys():
            await self.connected_channels[0].send(f'Character: {argv[1]} is an unknown character or does not have a model. Check playable characters with !characters')
            return
        if not argv[2] in stage_ids.keys():
            await self.connected_channels[0].send(f'Stage: {argv[2]} is an unknown. Check playable stages with !stages')
            return
        setting_up = time.time()
        await self.connected_channels[0].send(f'Setting matchup to {argv[0]} vs. {argv[1]} on {argv[2]}')
        environment_ip = "127.0.0.1"
        port = 20001
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"http://{environment_ip}:{port}/close_instance", headers=headers)
        os.killpg(os.getpgid(self.setupP1.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.setupP2.pid), signal.SIGTERM)
        self.setupP1 = subprocess.Popen([f"./display_latest.py {argv[0]} {self.char_model_dir[argv[0]]} {argv[2]}"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        self.setupP2 = subprocess.Popen([f"./display_latest.py {argv[1]} {self.char_model_dir[argv[1]]} {argv[2]}"], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)


    @commands.command()
    async def characters(self, ctx: commands.Context, *argv):
        o_str = "List of playable characters: "
        for char in self.char_model_dir.keys():
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