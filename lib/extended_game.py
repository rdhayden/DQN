from vizdoom import DoomGame
from vizdoom import Button
import numpy as np
import itertools as it

class ExtendedGame(DoomGame):
    def __init__(self, config_file_path):
        super(ExtendedGame, self).__init__()
        self.load_config(config_file_path)
        
        # TODO generalise this for later scenarios, for now explicitly set action combinations
        # combinations to try given that actions are always repeated we need to move and attack
        # at the same time but not attack standing still or it'll often waste bullets in the 
        # basic scenario:
        # MOVE_LEFT [1,0,0]
        # MOVE_RIGHT [0,1,0]
        # MOVE_LEFT + ATTACK [1,0,1]
        # MOVE_RIGHT + ATTACK [0,1,1]
        self.actions = [
            [1,0,0],
            [0,1,0],
            [1,0,1],
            [0,1,1]
        ]

        self.action_space = np.arange(len(self.actions)).tolist()
        self.n_actions = len(self.action_space)

    def sample_actions(self):
        idx = np.random.choice(self.n_actions)
        return self.action_space[idx]
    
    def make_action_int(self, action_int, repeat_for_frames):
        '''a utility method to make it possible to make actions with integers'''
        return self.make_action(self.actions[action_int], repeat_for_frames)