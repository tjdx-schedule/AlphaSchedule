# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""
from operator import itemgetter
import numpy as np

from utils import minKIndex2,Logger
from venvs.EnvirConf import envConfig 

EPSILON = 0.40
DIRNOISE = 0.1
IS_ADD_NOIRSE = False

class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_p, act =[]):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._P = prior_p
        self._Grade_Q = 0
        
        if not isinstance(act,list):
            act = [act]
        if parent is not None:
            self.act_seq = parent.act_seq + act
        else:
            self.act_seq = act
    
    def clear(self,keep_children_node=False):
        self._parent = None
        self._n_visits = 0
        self._P = 1.0
        self._Grade_Q = 0
        if not keep_children_node:
            self._children = {}
            
    def expand(self, action_priors,add_noise):
        if add_noise:
            action_priors = list(action_priors)
            length = len(action_priors)
            dirichlet_noise = np.random.dirichlet(DIRNOISE * np.ones(length))
            for i in range(length):
                if action_priors[i][0] not in self._children:
                    self._children[action_priors[i][0]] = \
                    TreeNode(self,(1-EPSILON)*action_priors[i][1]+EPSILON*dirichlet_noise[i],action_priors[i][0])
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob, action)

    def select(self, c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, grade):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Update Q, a running average of values for all visits.
        self._Grade_Q = grade

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, logger):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
              
        self.logger = logger
        
    def set_root_node(self,root):
        self._root = root

    def get_move_probs(self, game, K):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        self.hard_expand(game,K)

    def hard_expand(self,game,K):
        root = self._root
        if root.is_leaf():
            state = game.board.node_board(root)
            end, _ = state.game_end()
            if not end:
                action_probs = self._policy_prob(state,K)
                root.expand(action_probs, IS_ADD_NOIRSE)
                
                if K <= game.beam_size:            
                    obs_tuple = game.node_reset(root)
                    grades,act_seqs = game.evaluate_rollout(obs_tuple)
                    for index, dict_node in enumerate(root._children.items()):
                        act, node = dict_node
                        node.update(grades[index])
                    self.logger.update_arr(grades,act_seqs)
                else:
                    for act,node in root._children.items():
                        state_copy = game.board.node_board(node)
                        grade = self._evaluate_rollout(state_copy)
                        node.update(grade)    
                    
    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for i in range(limit):
            end, grade = state.game_end()
            if end:
                break
            action_probs = self._policy_prob(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.step(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        self.logger.update(grade,state.act_seq_last)
        return grade    

    def _policy_prob(self, state, K = None):
        current_state = state.canvas_obs
        legal_actions = state.availables
        
        _, action_mask = state.getActionMask()
        probs, _ = self._policy(current_state, action_mask)
        
        probs = probs[legal_actions]
        action_probs = zip(legal_actions,probs)
        if K is not None:
            K = min(K,len(legal_actions))
            if IS_ADD_NOIRSE:
                dirichlet_noise = np.random.dirichlet(DIRNOISE * np.ones_like(probs))
                probs = (1-EPSILON)*probs+EPSILON*dirichlet_noise
            
            indexes = np.argsort(-probs)[:K]
            action_probs = zip(np.array(legal_actions)[indexes],probs[indexes])
        return action_probs

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):#undo:step search and policy rollout
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 beam_size, mode='mcts_policy'):
        self.logger = Logger()
        self.mcts = MCTS(policy_value_function, self.logger)

        self.mode = mode
        
        self.beam_size = beam_size
        self.max_step = envConfig.partNum - 1
        self.act_size = envConfig.partNum
        self.first_N = 1
         
        
    def reset_player(self):
        self.forest = [TreeNode(None, 1.0)]
        self.logger.reset()
        
        self.step_count = 0

    def beam_tree_grow(self,game):
        grade_act_mat = np.inf * np.ones((len(self.forest),self.act_size))
        mat_counter = 0
        for i , tree in enumerate(self.forest):
            self.mcts.set_root_node(tree)
            dynamic_beam_size = self.act_size \
                if self.step_count < self.first_N else self.beam_size
            self.mcts.get_move_probs(game,dynamic_beam_size)
            for act,child_node in tree._children.items():
                grade_act_mat[i][act] = child_node._Grade_Q
                mat_counter += 1
        return [grade_act_mat,mat_counter]

    def create_new_forest(self, grade_act_mat_sta):
        indexes = minKIndex2(grade_act_mat_sta,self.beam_size,self.act_size)
        new_forest = []
        for x,y in indexes:
            newNode = self.forest[x]._children[y]
            newNode.clear()
            
            new_forest.append(newNode)
        self.forest = new_forest

    def mcts_search(self,game,is_sa_search = False):
        for i in range(self.max_step):
            # print(i)
            grade_act_mat_sta = self.beam_tree_grow(game)
            self.create_new_forest(grade_act_mat_sta)
        
            self.step_count += 1
        if is_sa_search:
            self.anneal(game)
        grade,_ = self.logger.getBest()
        return grade
    
    def anneal(self,game):
        grade_S1,S1 = self.logger.getBest()
        for i in range(self.beam_size):
            T = 1000*np.exp(-0.1*i*132/self.beam_size)
            for _ in range(self.beam_size):
                S2 = self.mutate(S1)
                S1,grade_S1=self.metropolis(game.board,S1,S2,T,grade_S1)
                self.logger.update(grade_S1,S1)
    
    def metropolis(self,board,S1,S2,T,grade_S1=None,grade_S2=None):
        grade1 = board.sche_seq(S1) if grade_S1 is None else grade_S1
        grade2 = board.sche_seq(S2) if grade_S2 is None else grade_S2
        dG = grade2 - grade1
        if dG < 0 or np.exp(-dG / T) >= np.random.rand():
            S = S2
            grade = grade2
        else:
            S = S1
            grade = grade1
        return S, grade

    def mutate(self,S1):
        N = len(S1)
        S2 = S1.copy()
        J1, J2 = np.random.choice(range(N), size=2, replace=False)
        W = S2[J1]
        S2[J1] = S2[J2]
        S2[J2] = W
        return S2
        
    def policy_rollout(self,board):
        done = False
        while not done:
            action_probs = self.mcts._policy_prob(board)
            acts, probs = zip(*action_probs)
            
            act_index = np.argmax(probs)
            move = acts[act_index]
            
            _, _, done, info = board.step(move)
        grade = info['episode']['r']
        return grade 
    
    def search(self, game):
        if self.mode == 'mcts_policy':
            grade = self.mcts_search(game,False)
        elif self.mode == 'pure_policy':
            grade = self.policy_rollout(game.board)
        else:
            raise Exception('policy mode error!')
        return grade
    
    def __str__(self):
        return "MCTS {}".format(self.player)

