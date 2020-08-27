# -*- coding: utf-8 -*-
import time
from venvs.excel import ExcelLog
from venvs.game import Game
from mcts_policy import MCTSPlayer

from policy_value_net_pytorch import PolicyValueNet

if __name__ == '__main__':
    start = time.time()    
    policy_mode = 'pure_policy'
    log_name = policy_mode + '_K2-35-15h'
    
    n_games = 100
    beam_size = 10
    
    logger = ExcelLog(log_name,True)
    path = './models/weight.model'
    update_net_model = PolicyValueNet(use_gpu=False,
                                      is_train = False)
    
    update_net_model.load_pretrained_weight(path,load_way='g2c')
    mcts_player = MCTSPlayer(update_net_model.policy_value_fn,
                             beam_size = beam_size,
                             mode = policy_mode)
    
    game = Game(update_net_model,beam_size,
                mode ='test',search_mode = policy_mode)
    grade_list = []
    for i in range(n_games):
        grade = game.start_play(mcts_player,
                                is_shown=0)
        grade_list.append(grade)
        print('n_play=',i+1,' grade:',grade)
        logger.saveTest(grade)
    average = sum(grade_list)/n_games
    print("num_playouts:{}, min: {}, average: {}, max:{}".format(
           100, min(grade_list), 
           average, max(grade_list)))    
    end = time.time()
    print("Execution Time: ", end - start)