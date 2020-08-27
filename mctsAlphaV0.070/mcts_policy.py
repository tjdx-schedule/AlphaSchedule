# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter

from utils import minKIndex2,Logger
from stateobs import VecNorm
from config import scheConfig 

EPSILON = 0.40
DIRNOISE = 0.1
IS_ADD_NOIRSE = 1

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)

class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_p, act =[]):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
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
        self._Q = 0
        self._u = 0
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


    def update(self, leaf_value, grade):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        self._Grade_Q += 1.0*(grade - self._Grade_Q) / self._n_visits

    def update_recursive(self, leaf_value, grade):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value, grade)
        self.update(leaf_value, grade)

    def get_value(self, c_puct):
        '''
        Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, vec_norm, policy_value_fn, logger, c_puct=5):
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
        
        self._c_puct = c_puct
        
        self._vec_norm = vec_norm
        self.logger = logger
        
    def set_root_node(self,root):
        self._root = root

    def get_move_probs(self, state, K, rollout_way='random'):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        self.hard_expand(state,K,rollout_way)

    def hard_expand(self,state,K,rollout_way):
        root = self._root
        if root.is_leaf():
            state = state.node_board(root)
            end, _ = state.game_end()
            if not end:
                action_probs = self._policy_prob(state,K)
                root.expand(action_probs, IS_ADD_NOIRSE)
                for act,node in root._children.items():
                    state_copy = state.node_board(node)
                    leaf_value, grade = self._evaluate_rollout(state_copy,rollout_way)
                    node.update_recursive(leaf_value, grade)          
    
    def _evaluate_rollout(self, state, rollout_way='random', limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for i in range(limit):
            end, grade = state.game_end()
            if end:
                break
            if rollout_way == 'random':
                action_probs = rollout_policy_fn(state)
            elif rollout_way == 'policy':
                action_probs = self._policy_prob(state)
            else:
                raise('rollout way error!')
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        winner = self._vec_norm._refilt(grade)
        self.logger.update(grade)
        return winner,grade

    def _policy_prob(self, state, K = None):
        current_state = self._vec_norm.ob_norm(state.current_state())
        legal_actions = state.availables
        
        probs, _ = self._policy(current_state, legal_actions)
        action_probs = zip(legal_actions,probs)
        if K is not None:
            K = min(K,len(legal_actions))
            
            indexes = np.argsort(-probs)[:K]
            action_probs = zip(np.array(legal_actions)[indexes],probs[indexes])
        return action_probs

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):#undo:step search and policy rollout
    """AI player based on MCTS"""

    def __init__(self, vec_norm, policy_value_function,
                 beam_size, c_puct=5, n_playout=2000, mode='mcts_policy'):
        if vec_norm is None:
            self.vec_norm = VecNorm()
        else:
            self.vec_norm = vec_norm
        
        self.logger = Logger()
        self.mcts = MCTS(self.vec_norm, policy_value_function, \
                         self.logger, c_puct)

        self.mode = mode
        
        self.beam_size = beam_size
        self.max_step = scheConfig.partNum - 1
        self.act_size = scheConfig.partNum
        self.first_N = 1
         
        
    def reset_player(self):
        self.forest = [TreeNode(None, 1.0)]
        self.logger.reset()
        
        self.step_count = 0

    def beam_tree_grow(self,board):
        grade_act_mat = np.inf * np.ones((len(self.forest),self.act_size))
        mat_counter = 0
        for i , tree in enumerate(self.forest):
            self.mcts.set_root_node(tree)
            if self.step_count < self.first_N:
                self.mcts.get_move_probs(board,self.act_size,'policy')
            else:
                self.mcts.get_move_probs(board,self.beam_size,'policy')
            
            for act,child_node in tree._children.items():
                grade_act_mat[i][act] = child_node._Grade_Q
                mat_counter += 1
        return [grade_act_mat,mat_counter]

    def create_new_forest(self, grade_act_mat_sta):
        indexes = minKIndex2(grade_act_mat_sta,self.beam_size,self.act_size)
        new_forest = []
        for x,y in indexes:
            newNode = self.forest[x]._children[y]
            print(self.step_count, newNode.act_seq, newNode._Grade_Q)
            newNode.clear()
            
            new_forest.append(newNode)
        self.forest = new_forest

    def mcts_search(self,board):
        for i in range(self.max_step):
            # print(i)
            grade_act_mat_sta = self.beam_tree_grow(board)
            self.create_new_forest(grade_act_mat_sta)
            print(self.step_count, 'logger:', self.logger.getBest())
            self.step_count += 1
        grade = self.logger.getBest()
        return grade
    
    def policy_rollout(self,board):
        done = False
        while not done:
            action_probs = self.mcts._policy_prob(board)
            acts, probs = zip(*action_probs)
            
            act_index = np.argmax(probs)
            move = acts[act_index]
            
            board.do_move(move)
            done, grade = board.game_end()
        return grade 
    
    def search(self, board):
        if self.mode == 'mcts_policy':
            grade = self.mcts_search(board)
        elif self.mode == 'pure_policy':
            grade = self.policy_rollout(board)
        else:
            raise Exception('policy mode error!')
        return grade
    
    def __str__(self):
        return "MCTS {}".format(self.player)

