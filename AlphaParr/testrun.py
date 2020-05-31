# -*- coding: utf-8 -*-
import time

from mcts_pure import MCTSPlayer as MCTS_Pure
from scheboard import  Game

if __name__ == '__main__':
    start = time.time()
    n_games = 10
    n_playout = 200
    mcts_player = MCTS_Pure(c_puct=5, n_playout=n_playout)
    game = Game()
    grade_list = []
    for i in range(n_games):
        grade = game.start_play(mcts_player,'val',0)
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
    

