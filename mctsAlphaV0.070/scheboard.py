# -*- coding: utf-8 -*-
import numpy as np
from parallel import Scheduler
from stateobs import StateObserve,VecNorm

class ScheBoard:
    def __init__(self, mode, seed):
        self.scheduler = Scheduler(mode, seed)
        self.obsDe = StateObserve(self.scheduler)
        
        self.action_size = self.scheduler.genpart.partNum
        
    def init_board(self,update_example=True):
        self.obsDe.pastStateInit()
        self.scheduler.reset(update_example)
        self.availables = self.get_legal()
        
    def node_board(self,treeNode):
        actSeq = treeNode.act_seq
        self.init_board(False)
        for act in actSeq:
            self.do_move(act)
        return self
    
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
    def __init__(self, board = None, mode ='train', seed=0):
        if board is None:
            self.board = ScheBoard(mode, seed)
        else:
            self.board = board
    
    def start_play(self, player, is_shown = 0):
        self.board.init_board()
        player.reset_player()
        grade = player.search(self.board)
        return grade

