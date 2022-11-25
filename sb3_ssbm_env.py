import gym
import random
from ssbm_gym.ssbm_env import SSBMEnv
from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.spaces import DiagonalActionSpace
from ssbm_gym.embed import EmbedGame, numActions, numCharacters, numStages
from ssbm_gym.gen_code import char_ids


COMPETITVE_STAGES = ['fod', 'stadium', 'yoshis_story', 'dream_land', 'battlefield', 'final_destination']
CHARACTERS = [name for name in char_ids.keys()]


class CustomEnv(SSBMEnv):
    def __init__(self, **kwargs):
        SSBMEnv.__init__(self, **kwargs)
        spaces = {
            'player0_character': gym.spaces.Box(low=-1, high=1, shape=(numCharacters,)),
            'player0_action_state': gym.spaces.Box(low=-1, high=1, shape=(numActions,)),
            'player0_state': gym.spaces.Box(low=-1, high=1, shape=(10,)),
            'player1_character': gym.spaces.Box(low=-1, high=1, shape=(numCharacters,)),
            'player1_action_state': gym.spaces.Box(low=-1, high=1, shape=(numActions,)),
            'player1_state': gym.spaces.Box(low=-1, high=1, shape=(10,)),
            'stage': gym.spaces.Box(low=-1, high=1, shape=(numStages,)),
        }
        self.dict_space = gym.spaces.Dict(spaces)
        self._embed_obs = EmbedGame(flat=False)
        self.metadata = {}
        self.custom_action = DiagonalActionSpace()
        #print(self.custom_action.n)

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = self.dict_space
            return self.dict_space

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            self._action_space = gym.spaces.Discrete(self.custom_action.n)
            return self._action_space

    def reset(self):
        # try:
        #     self.api.close()
        # except Exception as e:
        #     print(f"ERROR: {str(e)}")
        # options = dict(
        #     render=False,
        #     player1='ai',
        #     player2='cpu',
        #     char1='ics',
        #     char2=CHARACTERS[random.randint(0, len(CHARACTERS) - 1)],
        #     cpu2=7,
        #     stage=COMPETITVE_STAGES[random.randint(0, len(COMPETITVE_STAGES) - 1)],
        # )
        # self.api = DolphinAPI(**options)
        self.obs = self.api.reset()
        return self.embed_obs(self.obs)

    def embed_obs(self, obs):
        obs = self._embed_obs(obs)
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
