# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy

from stateobs import VecNorm

EPSILON = 0.20
DIRNOISE = 0.3

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        children_num = len(self._children)

        noise = np.random.dirichlet([DIRNOISE]*children_num)
        value_mat = np.zeros(children_num,dtype=np.float)
        nodes = list(self._children.items())
        for i, act_node in enumerate(nodes):
            node = act_node[1]
            _p = (1-EPSILON)*node._P + EPSILON*noise[i]
            node._u = (c_puct * _p * np.sqrt(node._parent._n_visits) / (1 + node._n_visits))
            value_mat[i] = node._Q + node._u
        return nodes[np.argmax(value_mat)]

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, vec_norm, policy_value_fn, c_puct=5, n_playout=10000):
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
        self._n_playout = n_playout
        self._vec_norm = vec_norm
        

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        current_state = self._vec_norm.ob_norm(state.current_state())
        legal_actions = state.availables
        action_probs, leaf_value = self._policy(current_state, legal_actions)
        
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            leaf_value = self._vec_norm._refilt(winner)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, vec_norm, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        if vec_norm is None:
            self.vec_norm = VecNorm()
        else:
            self.vec_norm = vec_norm
        self.mcts = MCTS(self.vec_norm, policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.step_count = 0
        
    def reset_player(self):
        self.mcts.update_with_move(-1)
        self.step_count = 0

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.action_size)
        if len(sensible_moves) > 0:
            temp = 1 if self._is_selfplay else 1e-3
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
                
            if self._is_selfplay:
                self.mcts.update_with_move(move)
            else:
                self.mcts.update_with_move(-1)
            
            self.step_count += 1
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
    
    def grade_norm(self,grade):
        grade = self.vec_norm._refilt(grade)
        return grade
    
    def state_norm(self, obs):
        obs = self.vec_norm.ob_norm(obs)
        return obs

    def __str__(self):
        return "MCTS {}".format(self.player)
