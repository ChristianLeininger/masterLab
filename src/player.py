import random
import numpy as np
import torch
from utils import log_array, get_max_key

from models import EvalModel



class RandomPlayer(object):
    def __init__(self, name):
        super(RandomPlayer, self).__init__()
        self.name = name

    def choose_action(self, state, valid_actions):
        return random.choice(valid_actions)


class ValidRandomPlayer(object):
    def __init__(self, name):
        super(ValidRandomPlayer, self).__init__()
        self.name = name

    def choose_action(self, state, valid_actions):
        # if the first row is not empty, then the column is full
        #if np.any(state[0, :] != 0):
        #     valid_actions = [a for a in valid_actions if state[0, a] == 0]
            # import pdb; pdb.set_trace()
        valid_actions = [a for a in valid_actions if state[0, a] == 0]
        return random.choice(valid_actions)
    


class MinMaxPlayer(object):
    def __init__(self, name, win_conditon=3, depth=3, player_number=1):
        super(MinMaxPlayer, self).__init__()
        self.name = name
        self.win_conditon = win_conditon
        self.depth = depth
        self.pn = player_number

    
    def choose_action(self, state, valid_actions):
        """ MinMaxPlayer chooses the action that maximizes the minimum
         
        Args:
            state (np.array): the current state of the game
        """
        # get the valid actions
        # best action
        dict_actions = {}
        for action in valid_actions:
            next_state = self.get_next_state(state=state, action=action, player_number=self.pn)
            value = self.evaluate_board_state(next_state)
            # get the minimax value for the next state
            # value = self.minimax(next_state, self.depth, False)
            # best_action.update({value: action})
            dict_actions.update({action: value})
        
        best_action = get_max_key(dict_actions)
        
        log_array(state)
        print(f"all actions: {dict_actions}")
        print(f"Player {self.pn} chooses column {best_action}")
        # get the best action
        import pdb; pdb.set_trace()
        return best_action
        

    def get_next_state(self, state, action, player_number):
        """ creates the next state of the game with the given action  
        
        Args:
            state (np.array): the current state of the game
            action (int): the action to take
        """
        # get the next state
        next_state = state.copy()
        # get the row to place the piece
        row = np.where(next_state[:, action] == 0)[0][-1]
        # place the piece
        next_state[row, action] = player_number
        return next_state


    def minimax(self, state, depth, maximizing_player):
        """ Minimax algorithm for finding the best move
        Args:
            state (np.array): the current state of the game
            depth (int): the depth of the search tree
            maximizing_player (bool): whether the current player is maximizing or not
        """
        # check if we are at a leaf node
        if depth == 0 or self.is_terminal(state):
            return self.evaluate_board_state(state)
        # if we are the maximizing player
        if maximizing_player:
            value = -np.inf
            # check all valid actions
            for action in self.get_valid_actions(state):
                # get the next state
                next_state = self.get_next_state(state, action)
                # recursively call minimax
                value = max(value, self.minimax(next_state, depth - 1, False))
            return value
        # if we are the minimizing player
        else:
            value = np.inf
            # check all valid actions
            for action in self.get_valid_actions(state):
                # get the next state
                next_state = self.get_next_state(state, action)
                # recursively call minimax
                value = min(value, self.minimax(next_state, depth - 1, True))
            return value

    

    def evaluate_board_state(self, state):
        """ Evaluate the board state for the current player
        Args:
            state (np.array): the current state of the game
        """
        score = 0
        # check if we would win
        if self.has_connect_n(state, self.win_conditon, self.pn):
            return 100
        # check if the opponent would win
        if self.has_connect_n(state, self.win_conditon, self.pn % 2 + 1):
            return -100
        # now check for various lengths of connections
        for i in range(2, self.win_conditon):
            # check if we have a connection of length i
            if self.has_connect_n(state, i, self.pn):
                score += 10
            # check if the opponent has a connection of length i
            if self.has_connect_n(state, i, self.pn % 2 + 1):
                score -= 10
        return score
    
    def has_connect_n(self, state, n, player_number):
        """ Check if the board state has a connection of length n for the given player 
        Args:
            state (np.array): the current state of the game
            n (int): the length of the connection
            player_number (int): the player number to check for
        """
        # check for horizontal connections
        for row in range(state.shape[0]):
            for col in range(state.shape[1] - n + 1):
                if np.all(state[row, col:col+n] == player_number):
                    return True
        # check for vertical connections
        for row in range(state.shape[0] - n + 1):
            for col in range(state.shape[1]):
                if np.all(state[row:row+n, col] == player_number):
                    return True
        # check for diagonal connections
        for row in range(state.shape[0] - n + 1):
            for col in range(state.shape[1] - n + 1):
                if np.all(np.diag(state[row:row+n, col:col+n]) == player_number):
                    return True
                if np.all(np.diag(np.fliplr(state[row:row+n, col:col+n])) == player_number):
                    return True
        return False


class RLAgent(object):
    def __init__(self, name, cfg, win_conditon=3, depth=3, player_number=1):
        super(RLAgent, self).__init__()
        self.name = name
        self.win_conditon = win_conditon
        self.depth = depth
        self.pn = player_number
        self.model = EvalModel(input_channels=1, 
                               conv_layers=[(1, 256, 3, 1, 1)], 
                               fc_layers=[64, 64]) 


    
    def choose_action(self, state, valid_actions):
        """ MinMaxPlayer chooses the action that maximizes the minimum
         
        Args:
            state (np.array): the current state of the game
        """
        # get the valid actions
        # best action
        dict_actions = {}
        for action in valid_actions:
            next_state = self.get_next_state(state=state, action=action, player_number=self.pn)
            value = self.evaluate_board_state(next_state)
            # get the minimax value for the next state
            # value = self.minimax(next_state, self.depth, False)
            # best_action.update({value: action})
            dict_actions.update({action: value})
        
        best_action = get_max_key(dict_actions)
        
        log_array(state)
        print(f"all actions: {dict_actions}")
        print(f"Player {self.pn} chooses column {best_action}")
        # get the best action
        import pdb; pdb.set_trace()
        return best_action
        

    def get_next_state(self, state, action, player_number):
        """ creates the next state of the game with the given action  
        
        Args:
            state (np.array): the current state of the game
            action (int): the action to take
        """
        # get the next state
        next_state = state.copy()
        # get the row to place the piece
        row = np.where(next_state[:, action] == 0)[0][-1]
        # place the piece
        next_state[row, action] = player_number
        return next_state


    def minimax(self, state, depth, maximizing_player):
        """ Minimax algorithm for finding the best move
        Args:
            state (np.array): the current state of the game
            depth (int): the depth of the search tree
            maximizing_player (bool): whether the current player is maximizing or not
        """
        # check if we are at a leaf node
        if depth == 0 or self.is_terminal(state):
            return self.evaluate_board_state(state)
        # if we are the maximizing player
        if maximizing_player:
            value = -np.inf
            # check all valid actions
            for action in self.get_valid_actions(state):
                # get the next state
                next_state = self.get_next_state(state, action)
                # recursively call minimax
                value = max(value, self.minimax(next_state, depth - 1, False))
            return value
        # if we are the minimizing player
        else:
            value = np.inf
            # check all valid actions
            for action in self.get_valid_actions(state):
                # get the next state
                next_state = self.get_next_state(state, action)
                # recursively call minimax
                value = min(value, self.minimax(next_state, depth - 1, True))
            return value

    

    def evaluate_board_state(self, state):
        """ Evaluate the board state for the current player
            Uses the neural network to evaluate the board state
        Args:
            state (np.array): the current state of the game
        """
        self.model.eval()
        # get the score
        
        score = self.model(torch.Tensor(state).unsqueeze(0))
        import pdb; pdb.set_trace()
        
        return 
    
    def has_connect_n(self, state, n, player_number):
        """ Check if the board state has a connection of length n for the given player 
        Args:
            state (np.array): the current state of the game
            n (int): the length of the connection
            player_number (int): the player number to check for
        """
        # check for horizontal connections
        for row in range(state.shape[0]):
            for col in range(state.shape[1] - n + 1):
                if np.all(state[row, col:col+n] == player_number):
                    return True
        # check for vertical connections
        for row in range(state.shape[0] - n + 1):
            for col in range(state.shape[1]):
                if np.all(state[row:row+n, col] == player_number):
                    return True
        # check for diagonal connections
        for row in range(state.shape[0] - n + 1):
            for col in range(state.shape[1] - n + 1):
                if np.all(np.diag(state[row:row+n, col:col+n]) == player_number):
                    return True
                if np.all(np.diag(np.fliplr(state[row:row+n, col:col+n])) == player_number):
                    return True
        return False

