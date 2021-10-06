import numpy as np

from .scheEnv import ScheEnv
from .envs import make_vec_envs

class Game:
    def __init__(self, model, beam_size, mode ='train', seed=0):
        self.beam_size = beam_size
        self.board = ScheEnv(seed,mode)
        self.envs = make_vec_envs(seed,mode,beam_size)
        self.model = model
    
    def start_play(self, player, is_shown = 0):
        self.board.init_board()
        self.envs.init_board()
        player.reset_player()
        grade = player.search(self)
        return grade
    
    def node_reset(self, root):
        k = self.envs.reset()
        print(k[5])
        root_act_len = len(root.act_seq)
        act_arr = -np.ones((self.beam_size,root_act_len+1))
        for index, dict_node in enumerate(root._children.items()):
            act, node = dict_node
            act_arr[index] = node.act_seq
        for i in range(root_act_len+1):
            obs,_,dones, infos = self.envs.step(act_arr[:,i])
            
        self.dynamic_beam_size = index+1
        return [obs,dones,infos]
            
    def evaluate_rollout(self, obs_tuple):
        [obs,dones,infos] = obs_tuple
        result = []
        while 1:
            if dones[0]:
                for i in range(self.dynamic_beam_size):
                    result.append(infos[i]['episode']['r'])
                break
            
            _,action_masks = self.envs.get_action_mask()
            acts = self.model.policy_value_acts(obs,action_masks)
            if self.dynamic_beam_size < self.beam_size:
                acts[self.dynamic_beam_size:] = -1
            
            obs,_,dones,infos = self.envs.step(acts)
            
      
    