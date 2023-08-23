import gym
from gym import spaces
import numpy as np
import logging

from player import RandomPlayer
from utils import set_seed, log_array
logger = logging.getLogger(__name__)



class ConnectFourEnv(gym.Env):
    def __init__(self,  player1, player2):
        super(ConnectFourEnv, self).__init__()

        # Define action and observation spaces
        self.space_size = [6, 7]
        self.action_space = spaces.Discrete(self.space_size[1]) # 7 columns to choose from
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.space_size[0], self.space_size[1]), dtype=np.int32) # 6 rows x 7 columns board
        self.players = [player1, player2]
        self.valid_actions = [a for a in range(self.space_size[1])]
        # Initialize state (board)
        self.reset()
    
    def reset(self):
        # Reset the state (board) to an empty state
        self.state = np.zeros((self.space_size[0], self.space_size[1]), dtype=np.int32)
        self.done = False
        self.steps = 0
        self.current_player_number = 0 
        return self.state

    def step(self, action):
        # Implement game logic here
        # ...
        # Update state (board)
        logger.info(f'Player {self.current_player_number} chose action {action}')
        # check if action is valid
        valid = self.check_valid_action(action)
        if not valid:
            logger.info(f'Player {self.current_player_number} chose invalid action {action}')
            return self.state, -1, True, {"false_action": True}
        # place token at the for agiven column
        done = self.place_token(action)

        # Return observation, reward, done, and info
        reward = 0
        
        return self.state, reward, done, {"false_action": "false"}

    def place_token(self, action):
        """ Places a token at the given column. 
        Args:
            action (int): column to place token
        """
        # assert column is valid
        assert self.check_valid_action(action)
        # find the first empty row in the column
        row = self.space_size[0] - 1      # rows count from bottom to top
        # import pdb; pdb.set_trace()
        while self.state[row][action] != 0:
            row -= 1
        # place token
        self.state[row, action] = self.current_player_number + 1
        logger.info(f'Placed token at row {row} and column {action}')
        # check if this action leads to victory
        # update valid actions
        self.valid_actions = [a for a in range(self.space_size[1]) if self.check_valid_action(a)]
        done = False
        if self.check_victory(row, action):
            done = True
            logger.info(f'Player {self.current_player_number} won!')
        return done


    def play(self):
        # Implement game logic here
        # ...
        logger.info(f'Starting game at {self.steps} and Player {self.current_player_number}')
        done  = False
        while not done:
            current_player = self.players[self.current_player_number]
            action = current_player.choose_action(self.state, self.valid_actions)
            action = 6
            self.state, reward, done, info = self.step(action)
            self.steps += 1
            logger.info(f'Player {self.current_player_number} chose action {action} and got reward {reward}')
            # logger.info(f'State: {self.state}')
            self.current_player_number = (self.current_player_number + 1) % 2
            if done:
                # import pdb; pdb.set_trace()
                if info["false_action"]:
                    logger.info(f'Game ended after {self.steps} steps with player {self.current_player_number} winning due to false action')
                    break
                logger.info(f'Game ended after {self.steps} steps with player {self.current_player_number} winning')
                log_array(self.state)
                break
                
            # check for draw 
            if not np.any(self.state == 0):
                done = True
                logger.info(f'Game ended in draw after {self.steps} steps')
                break
        # import pdb; pdb.set_trace()
        
        logger.info(f'Game ended after {self.steps} steps')

    
    def check_victory(self, row, col):
        player = self.state[row, col]

        # Check horizontally
        if self.check_direction(row, col, 0, 1, player) or \
        self.check_direction(row, col, 0, -1, player):
            return True

        # Check vertically
        if self.check_direction(row, col, 1, 0, player):
            return True

        # Check diagonally (top-left to bottom-right)
        if self.check_direction(row, col, 1, 1, player) or \
        self.check_direction(row, col, -1, -1, player):
            return True

        # Check diagonally (top-right to bottom-left)
        if self.check_direction(row, col, 1, -1, player) or \
        self.check_direction(row, col, -1, 1, player):
            return True

        return False

    def check_direction(self, row, col, d_row, d_col, player):
        consecutive_count = 0
        for i in range(-3, 4):
            r = row + i * d_row
            c = col + i * d_col
            if 0 <= r < 6 and 0 <= c < 7 and self.state[r, c] == player:
                consecutive_count += 1
                if consecutive_count == 4:
                    return True
            else:
                consecutive_count = 0
        return False
        

    def check_valid_action(self, action):
        # check if in this column there is still space
        if self.state[0, action] != 0:
            return False
        return True
        
    def render(self, mode='human'):
        # Implement rendering logic here
        # ...
        pass






if __name__ == '__main__':
    # Create two random players
    seed = 42
    set_seed(seed)
    player1 = RandomPlayer('Player 1')
    player2 = RandomPlayer('Player 2')
    # initialize logger
    logging.basicConfig(level=logging.INFO)
    # Initialize environment
    env = ConnectFourEnv(player1=player1, player2=player2)
    env.play()