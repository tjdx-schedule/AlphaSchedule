# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import time
import argparse
import numpy as np
from collections import deque

from scheboard import ScheBoard,Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from stateobs import VecNorm
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from utils import save_buffer, load_buffer
from excel import ExcelLog
from config import scheConfig

def get_args():
    parser = argparse.ArgumentParser(description='AL')
    parser.add_argument(
        '--load',
        action='store_true',
        default=False,
        help='load trained some model to train more')
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False,
        help='load trained some model to train more')
    args = parser.parse_args()
    return args

class TrainPipeline():
    def __init__(self, args):
        # params of the board and the game
        self.board = ScheBoard()
        self.game = Game(self.board)
        self.path = './models/'
        self.best_net_name = self.path + 'best.model'
        self.update_net_name = self.path + 'current.model'
        # training params
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move
        self.c_puct = 1.5
        self.buffer_size = 100000
        self.batch_size = 256  # mini-batch size for training 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 4  # num of train_steps for each update
        self.kl_targ = 0.02
        
        self.check_freq = 50
        self.best_win_ratio = 1e5
        self.vec_norm = VecNorm()
        self.logger = ExcelLog(isLog=True)
        
        self.N = scheConfig.period
        self.train_grade = deque(maxlen=self.N)
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        if args.load:
            # start training from an initial policy-value net
            print('load Model success!')
            self.best_net = PolicyValueNet(model_file=self.best_net_name,use_gpu=args.cuda)
            self.update_net = PolicyValueNet(model_file=self.update_net_name,use_gpu=args.cuda)
            self.data_buffer = load_buffer(self.data_buffer)
            
            self.best_win_ratio, self.vec_norm,self.logger = self.update_net.save_dict
            self.logger.setLog()
            print('best model win_ratio:', self.best_win_ratio)
        else:
            # start training from a new policy-value net
            self.logger = ExcelLog(isLog=True)
            self.best_net = PolicyValueNet(use_gpu=args.cuda)
            self.update_net = PolicyValueNet(use_gpu=args.cuda)
            self.best_net.update_model(self.update_net.get_policy_param())
        self.mcts_player = MCTSPlayer(self.vec_norm,
                                      self.update_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            # play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            self.train_grade.append(winner)
            
            self.average = sum(self.train_grade)/len(self.train_grade)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_first_batch, state_second_batch, mcts_probs_batch, winner_batch = [], [], [], []
        for state, probs, winner in  mini_batch:
            state_first_batch.append(state[0])
            state_second_batch.append(state[1])
            mcts_probs_batch.append(probs)
            winner_batch.append(winner)
        state_batch = [state_first_batch,state_second_batch]
        old_probs, old_v = self.update_net.policy_value(state_batch)
        for i in range(self.epochs):
            value_loss, policy_loss, loss, entropy \
            = self.update_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.update_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "loss:{:.4f},"
               "entropy:{:.4f},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        self.logger.saveLoss(float(kl), float(value_loss), float(policy_loss),
                             float(loss) , float(entropy),
                             float(explained_var_old), float(explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.vec_norm,
                                         self.update_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        grade_list = []
        for i in range(n_games):
            grade = self.game.start_play(current_mcts_player,
                                         'val', is_shown=0)
            grade_list.append(grade)
        average = sum(grade_list)/n_games
        print("num_playouts:{}, min: {}, average: {}, max:{}".format(
               self.n_playout, min(grade_list), 
               average, max(grade_list)))
        self.logger.saveTest(min(grade_list),average,max(grade_list))
        return average
    
    def run(self):
        """run the training pipeline"""
        try:
            i = 0
            start_time = time.time()
            while True:
                i += 1
                self.collect_selfplay_data(self.play_batch_size)
                print("Time {}, batch :{}, episode_len:{}, train_grade:{:.3f}".format(
                    time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),    
                    i, self.episode_len, self.average))
                self.logger.saveTrain(time.strftime\
                                      ("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)), 
                                      self.average)
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.update_net.save_model(self.update_net_name, 
                                               self.best_win_ratio,
                                               self.vec_norm)
                    if win_ratio <= self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.best_net.update_model(self.update_net.get_policy_param())
                        self.best_net.save_model(self.best_net_name)
        except KeyboardInterrupt:
            print('\n\rquit')
            self.update_net.save_model(self.update_net_name, 
                                       self.best_win_ratio,
                                       self.vec_norm,
                                       self.logger)
            self.best_net.save_model(self.best_net_name)
            save_buffer(self.data_buffer)



def main():
    args = get_args()
    training_pipeline = TrainPipeline(args)
    training_pipeline.run()

if __name__ == '__main__':
    main()
