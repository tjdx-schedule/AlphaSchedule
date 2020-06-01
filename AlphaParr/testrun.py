# -*- coding: utf-8 -*-
import time

from mcts_pure import MCTSPlayer as MCTS_Pure
from scheboard import  Game
from excel import ExcelLog

if __name__ == '__main__':
    start = time.time()
    n_games = 100
    n_playout = 215
    c_puct = 1.0
    logger = ExcelLog('mcts_atc',True)
    
    mcts_player = MCTS_Pure(c_puct=c_puct, n_playout=n_playout)
    game = Game()
    grade_list = []
    for i in range(n_games):
        grade = game.start_play(mcts_player,'test',0)
        
        logger.saveTest(grade)
        grade_list.append(grade)
        print('pure_n_play=',i+1,' grade:',grade)
        # print(game.board.scheduler.T)
    average = sum(grade_list)/n_games
    print("num_playouts:{}, min: {}, average: {}, max:{}".format(
       n_playout, min(grade_list), 
       average, max(grade_list)))
    
    # treeLogDict = mcts_player.treeLog._outputDict()
    # mcts_player.treeLog.plotTree()
    
    end = time.time()
    print("Execution Time: ", end - start)
    

