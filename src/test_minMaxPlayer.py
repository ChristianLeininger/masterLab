import unittest
import numpy as np
from main import ConnectFourEnv
from player import MinMaxPlayer
from utils import log_array

class TestConnectFourEnv(unittest.TestCase):

    def setUp(self):
        self.player1 = MinMaxPlayer(name="MinMaxPlayer1",win_conditon=3, depth=3)
        board = np.zeros((4,4), dtype=int)

    def test_has_conntect_n_1(self):
        board = np.zeros((4,4), dtype=int)
        board[0,0] = 1
        board[0,1] = 1
        board[0,2] = 1
        self.assertTrue(self.player1.has_connect_n(board, 3, 1))
        self.assertFalse(self.player1.has_connect_n(board, 4, 1))
        self.assertFalse(self.player1.has_connect_n(board, 3, 2))
    
    def test_has_connect_n_2(self):
        board = np.zeros((4,4), dtype=int)
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        self.assertTrue(self.player1.has_connect_n(board, 3, 1))
        self.assertFalse(self.player1.has_connect_n(board, 4, 1))
        self.assertFalse(self.player1.has_connect_n(board, 3, 2))

    def test_has_connect_n_3(self):
        board = np.zeros((4,4), dtype=int)
        board[0,0] = 1
        board[1,0] = 1
        board[2,0] = 1
        self.assertTrue(self.player1.has_connect_n(board, 3, 1))
        self.assertFalse(self.player1.has_connect_n(board, 4, 1))
        self.assertFalse(self.player1.has_connect_n(board, 3, 2))
    

    def test_evaluate_board_state_1(self):
        board = np.zeros((4, 4), dtype=int)
        self.player1.win_conditon = 3
        board[0,0] = 1
        board[0,1] = 1
        board[0,2] = 1
        self.assertEqual(self.player1.evaluate_board_state(board), 100)
    
    def test_evaluate_board_state_2(self):
        board = np.zeros((4, 4), dtype=int)
        self.player1.win_conditon = 3
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        self.assertEqual(self.player1.evaluate_board_state(board), 100)
    
    def test_evaluate_board_state_3(self):
        board = np.zeros((4, 4), dtype=int)
        self.player1.win_conditon = 3
        board[0,0] = 2
        board[1,0] = 2
        board[2,0] = 2
        self.assertEqual(self.player1.evaluate_board_state(board), -100)
    
    def test_evaluate_board_state_4(self):
        board = np.zeros((4, 4), dtype=int)
        self.player1.win_conditon = 4
        board[0,0] = 2
        board[1,1] = 2
        board[2,2] = 2
        self.assertEqual(self.player1.evaluate_board_state(board), -20)
    
    def test_get_next_state(self):
        board = np.zeros((4, 4), dtype=int)
        board[3,3] = 2
        board[3,2] = 2
        board[2,2] = 2
        next_state = self.player1.get_next_state(board, action=0, player_number=1)
        self.assertEqual(next_state[3,3], 2)
        self.assertEqual(next_state[3,2], 2)
        self.assertEqual(next_state[2,2], 2)
        self.assertEqual(next_state[3,0], 1)
    
    def test_get_next_state_2(self):
        board = np.zeros((4, 4), dtype=int)
        board[3,3] = 2
        board[3,2] = 2
        board[2,2] = 2
        next_state = self.player1.get_next_state(board, action=1, player_number=1)
        self.assertEqual(next_state[3,3], 2)
        self.assertEqual(next_state[3,2], 2)
        self.assertEqual(next_state[2,2], 2)
        self.assertEqual(next_state[3,1], 1)

    
if __name__ == "__main__":
    unittest.main()
