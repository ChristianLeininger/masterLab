import unittest
from main import ConnectFourEnv
from player import RandomPlayer
from utils import log_array

class TestConnectFourEnv(unittest.TestCase):

    def setUp(self):
        player1 = RandomPlayer('Player 1')
        player2 = RandomPlayer('Player 2')
        self.env = ConnectFourEnv(player1=player1, player2=player2)
    
    def test_vertical_victory_bottom(self):
        self.env.state[2:7, 2] = 1
        self.assertTrue(self.env.check_vertical_victory(col=2, player=1))
    
    def test_vertical_victory_middle(self):
        # clear the board
        self.env.state = self.env.state * 0
        self.env.state[1:5, 3] = 1
        self.assertTrue(self.env.check_vertical_victory(col=3, player=1))
    
    def test_vertical_victory_top(self):
        self.env.state = self.env.state * 0
        self.env.state[0:4, 4] = 1
        self.assertTrue(self.env.check_vertical_victory(col=4, player=1))

    def test_horizontal_victory_start(self):
        self.env.state = self.env.state * 0
        self.env.state[3, 0:4] = 1
        self.assertTrue(self.env.check_horizontal_victory(row=3, player=1))
    
    def test_horizontal_victory_middle(self):
        self.env.state = self.env.state * 0
        self.env.state[2, 1:5] = 1
        self.assertTrue(self.env.check_horizontal_victory(row=2, player=1))
    
    def test_horizontal_victory_end(self):
        self.env.state = self.env.state * 0
        self.env.state[1, 3:7] = 1
        # print(self.env.state)
        self.assertTrue(self.env.check_horizontal_victory(row=1, player=1))
    
    def test_horizontal_victory_false(self):
        self.env.state = self.env.state * 0
        self.env.state[1, 2:7] = 1
        self.env.state[1, 5] = 2
        self.assertFalse(self.env.check_horizontal_victory(row=1, player=1))


if __name__ == "__main__":
    unittest.main()
