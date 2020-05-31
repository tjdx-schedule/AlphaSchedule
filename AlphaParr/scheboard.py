# -*- coding: utf-8 -*-
import numpy as np
from parallel import Scheduler
from stateobs import StateObserve,VecNorm

class ScheBoard:
    def __init__(self):
        self.scheduler = Scheduler()
        self.obsDe = StateObserve(self.scheduler)
        
        self.action_size = self.scheduler.genpart.partNum
        
    def init_board(self, mode='train'):
        self.scheduler.reset(mode)
        self.availables = self.get_legal()
    
    def current_state(self):
        obs = self.obsDe.observation()
        return obs
    
    def do_move(self, move):
        if move not in self.availables:
            raise Exception('not legal act')
        self.scheduler.step(move)
        self.availables = self.get_legal()
    
    def game_end(self):#undo:grade norm
        done, grade = self.scheduler.is_end()
        return done,grade
    
    def get_legal(self):
        return self.scheduler.available()
    
    def show_gantt(self):
        self.scheduler.plotGantt()
    

class Game:
    def __init__(self, board = None):
        if board is None:
            self.board = ScheBoard()
        else:
            self.board = board
    
    def start_play(self, player, mode = 'train', is_shown = 0):
        self.board.init_board(mode)
        player.reset_player()
        while True:
            move = player.get_action(self.board)
            self.board.do_move(move)
            end, grade = self.board.game_end()
            if end:
                break        
        if is_shown:
            self.board.show_gantt()
        return grade
    
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.init_board()
        player.reset_player()
        states, mcts_probs = [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(player.state_norm(self.board.current_state()))
            mcts_probs.append(move_probs)
            # perform a move
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                grade = player.grade_norm(winner)
                # winner from the perspective of the current player of each state
                winners_z = grade*np.ones(len(states))
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    self.board.show_gantt()
                return winner, zip(states, mcts_probs, winners_z)            

if __name__ == '__main__':
    a = ScheBoard()    
    norm = VecNorm()
    
    for i in range(5):
        actLi = []
        stateLi = []
        a.init_board()
        done,_ = a.game_end()
        while not done:
            act = np.random.choice(a.availables)
            actLi.append(act)
            stateLi.append(norm.ob_norm(a.current_state()))
            a.do_move(act)
            done, grade = a.game_end()
        print(grade)
