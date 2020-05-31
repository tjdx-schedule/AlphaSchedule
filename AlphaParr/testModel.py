# -*- coding: utf-8 -*-
import time
from scheboard import Game
from mcts_alphaZero import MCTSPlayer
from stateobs import VecNorm
from policy_value_net_pytorch import PolicyValueNet
from excel import ExcelLog

if __name__ == '__main__':
    start = time.time()    
    logger = ExcelLog('bestNet',True)
    
    n_games = 100
    n_playout = 200
    c_puct = 1.5

    path = './models/'
    best_net_name = path + 'best.model'
    update_net_name = path + 'current.model'
    best_net = PolicyValueNet(model_file=best_net_name,use_gpu=True)
    update_net = PolicyValueNet(model_file=update_net_name,use_gpu=True)
    
    best_ratio, vec_norm, _ = update_net.save_dict
    print('best model win_ratio:', best_ratio)
    
    mcts_player = MCTSPlayer(vec_norm,
                             best_net.policy_value_fn,
                             c_puct=c_puct,
                             n_playout=n_playout)
    
    game = Game()
    grade_list = []
    for i in range(n_games):
        grade = game.start_play(mcts_player,
                                'test',
                                is_shown=0)
        logger.saveTest(grade)
        grade_list.append(grade)
        print('n_play=',i+1,' grade:',grade)
    average = sum(grade_list)/n_games
    print("num_playouts:{}, min: {}, average: {}, max:{}".format(
           n_playout, min(grade_list), 
           average, max(grade_list)))    
    end = time.time()
    print("Execution Time: ", end - start)