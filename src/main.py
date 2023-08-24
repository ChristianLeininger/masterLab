import gym
from gym import spaces
import numpy as np
import logging

from player import RandomPlayer, ValidRandomPlayer
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
        self.total_eposides = 1000
        # Initialize state (board)
        self.reset()
    
    def reset(self):
        # Reset the state (board) to an empty state
        self.state = np.zeros((self.space_size[0], self.space_size[1]), dtype=np.int32)
        self.done = False
        self.steps = 0
        self.current_player_number = 0
        self.valid_actions = [a for a in range(self.space_size[1])]
        return self.state

    def step(self, action):
        # Implement game logic here
        # ...
        # Update state (board)
        logger.debug(f'Player {self.current_player_number} chose action {action}')
        # check if action is valid
        valid = self.check_valid_action(action)
        if not valid:
            logger.debug(f'Player {self.current_player_number} chose invalid action {action}')
            return self.state, -1, True, {"false_action": True}
        # place token at the for agiven column
        done = self.place_token(action)
        # Return observation, reward, done, and info
        reward = 0
        
        return self.state, reward, done, {"false_action": False}

    def place_token(self, action):
        """ Places a token at the given column. 
        Args:
            action (int): column to place token
        """
        # assert column is valid
        assert self.check_valid_action(action)
        # find the first empty row in the column
        row = self.space_size[0] - 1      # rows count from bottom to top
        while self.state[row][action] != 0:
            row -= 1
        # place token
        self.state[row, action] = self.current_player_number + 1
        logger.debug(f'Placed token at row {row} and column {action}')
        # check if this action leads to victory
        # update valid actions
        self.valid_actions = [a for a in range(self.space_size[1]) if self.check_valid_action(a)]
    
        return self.check_victory(row=row, col=action)
    
    def check_victory(self, row: int, col: int) -> bool:
        """ check all 4 possible victory conditions. 
        Args:
            row (int): current row
            col (int): current column
        """
        done = False
        player = self.state[row, col]
        # check vertical victory
        if self.check_vertical_victory(col=col, player=player):
            done = True
        if self.check_horizontal_victory(row=row, player=player):
            done = True
        if self.check_right_diagonal_victory(row=row, col=col, player=player):
            done = True
        if self.check_left_diagonal_victory(row=row, col=col, player=player):
            done = True
        return done
    
    def update_results(self, results, winner, steps, reward, info):
        """ Updates the results dictionary with the results of the current game.
        """
        
        if info == "draw":
            results["draw"] += 1
            # give both players a reward of 0
            results[self.players[0].name]["rewards"].append(0)
            results[self.players[1].name]["rewards"].append(0)
        else:
            results[self.players[winner].name]["wins"] += 1
            results[self.players[winner].name]["rewards"].append(reward)
        results["steps"].append(steps)
        results["episodes"].append(self.steps)
        return results
    
    def show_statistics(self, results):
        """Show statistics of the results of the games. 
        Args:
            results (dict): results of the games
        """
        logging.info(f'Player {self.players[0].name} won {results[self.players[0].name]["wins"]} games')
        logging.info(f'Player {self.players[1].name} won {results[self.players[1].name]["wins"]} games')
        logging.info(f'Number of draws: {results["draw"]}')
        logging.info(f'Average number of steps: {np.mean(results["steps"])}')
        logging.info(f'Average reward of {self.players[0].name}: {np.mean(results[self.players[0].name]["rewards"]):.2f}')
        logging.info(f'Average reward of {self.players[1].name}: {np.mean(results[self.players[1].name]["rewards"]):.2f}')

    def start_games(self, num_games):
        """ Starts a number of games and returns the results. 
        Args:
            num_games (int): number of games to play
        """
        results = {self.players[0].name: {"wins": 0, "rewards":[]}, self.players[1].name: {"wins": 0, "rewards":[]} , "draw": 0, "steps": [], "episodes": []}
        for episode in range(self.total_eposides):
            self.reset()
            winner, steps, reward, info = self.play(episode=episode)
            self.update_results(results, winner, steps, reward, info)
            if info == "draw":
                logging.info(f'Episode {episode} of {self.total_eposides} ended in draw after {steps} steps')
            else:
                logging.info(f'Episode {episode} of {self.total_eposides} ended after {steps} steps with player {winner} winning')

        self.show_statistics(results)
        

    def play(self, episode):
        """ play the game until a player wins or the board is full. 
        
        Args:
            episode (int): episode number
        """
        # logger.info(f'Starting  {episode} episode at {self.steps} and Players {self.current_player_number}')
        done  = False
        info = ""
        while not done:
            current_player = self.players[self.current_player_number]
            action = current_player.choose_action(self.state, self.valid_actions)
            self.state, reward, done, info = self.step(action)
            self.steps += 1
            #logger.info(f'Player {self.current_player_number} chose action {action} and got reward {reward}')
            # logger.info(f'State: {self.state}')
            self.current_player_number = (self.current_player_number + 1) % 2
            if done:
                if info["false_action"]:
                    # logger.info(f'Game ended after {self.steps} steps with player {self.current_player_number} winning due to false action')
                    info = "false_action"
                    break
                # logger.info(f'Game ended after {self.steps} steps with player {self.current_player_number} winning')
                info = "win"
                reward = 1
                break
            # check for draw 
            if not np.any(self.state[0] == 0):
                done = True
                info = "draw"
                #logger.info(f'Game ended in draw after {self.steps} steps')
                break
    
        # log_array(self.state)
        # logger.info(f'Game ended after {self.steps} steps')
        return self.current_player_number, self.steps, reward, info 
    
    def check_victory(self, row, col):
        player = self.state[row, col]
        # First case - vertical victory
        # check the current column for 4 consecutive tokens of the same player
        if self.check_vertical_victory(col, player):
            logger.debug(f'Player {player} won with vertical colmn {col}')
            return True
        return False

    
    def check_vertical_victory(self, col: int, player: int):
        """ Checks if the current player has 4 consecutive tokens in the given column.
    
        Args:
            col (int): current column
            player (int): current player 
        """
        # start from the bottom of the column
        row = self.space_size[0] - 1
        count = 0
        while row >= 0:
            if self.state[row, col] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True
            row -= 1
        return False
    
    def check_horizontal_victory(self, row: int, player: int):
        """ Checks if the current player has 4 consecutive tokens in the given row.
    
        Args:
            row (int): current row
            player (int): current player 
        """
        # start from the bottom of the column
        col = 0
        count = 0
        while col < self.space_size[1]:
            if self.state[row, col] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True
            col += 1
        return False
    
    def check_right_diagonal_victory(self, row: int, col: int, player: int):
        """ Checks if the current player has 4 consecutive tokens in the given row. 
        
        Args:
            row (int): current row
            col (int): current column
            player (int): current player 
        """
        # start from the bottom of the column
        count = 0
        while col < self.space_size[1] and row < self.space_size[0]:
            if self.state[row, col] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True
            col += 1
            row += 1
        return False
    
    def check_left_diagonal_victory(self, row: int, col: int, player: int):
        """ Checks if the current player has 4 consecutive tokens in the given row. 
        
        Args:
            row (int): current row
            col (int): current column
            player (int): current player 
        """
        # start from the bottom of the column
        count = 0
        while col >= 0 and row < self.space_size[0]:
            if self.state[row, col] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True
            col -= 1
            row += 1
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
    player1 = ValidRandomPlayer('Valid random Player 1')
    player2 = ValidRandomPlayer('Valid random Player 2')
    # initialize logger
    logging.basicConfig(level=logging.INFO)
    # Initialize environment
    env = ConnectFourEnv(player1=player1, player2=player2)
    env.start_games(num_games=100)