# -*- coding: utf-8 -*-
import time
from excel import ExcelLog
from scheboard import Game,VecNorm
from mcts_policy import MCTSPlayer

from policy_value_net_pytorch import PolicyValueNet

if __name__ == '__main__':
    start = time.time()    
    policy_mode = 'pure_policy'
    log_name = policy_mode + '_6530'
    
    n_playout = 250
    c_puct = 0.5
    n_games = 100
    
    logger = ExcelLog(log_name,True)
    path = './models/weight.model'
    update_net_model = PolicyValueNet(use_gpu=False,
                                      is_train = False)
    
    vec_norm = VecNorm()
    update_net_model.load_pretrained_weight(path,vec_norm,load_way='g2c')
    mcts_player = MCTSPlayer(vec_norm,
                             update_net_model.policy_value_fn,
                             beam_size = 1,
                             c_puct=c_puct,
                             n_playout=n_playout,
                             mode = policy_mode)
    
    game = Game(mode ='test')
    grade_list = []
    for i in range(n_games):
        grade = game.start_play(mcts_player,
                                is_shown=0)
        grade_list.append(grade)
        print('n_play=',i+1,' grade:',grade)
        logger.saveTest(grade)
    average = sum(grade_list)/n_games
    print("num_playouts:{}, min: {}, average: {}, max:{}".format(
           n_playout, min(grade_list), 
           average, max(grade_list)))    
    end = time.time()
    print("Execution Time: ", end - start)