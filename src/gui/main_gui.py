import os
import sys
import tkinter as tk
import numpy as np
import logging
from PIL import Image, ImageDraw

class ConnectFour:
    def __init__(self, root, logger):
        self.logger = logger
        self.blink_color = 'white'
        self.blinking = False
        self.root = root
        self.rows, self.columns = 6, 7
        self.field_size = 50
        self.current_player = 1
        self.player_colors = {1: 'red', 2: 'yellow'}
        if self.root is not None:
            self.root.title("Connect 4")
            self.screen_width = self.root.winfo_screenwidth()
            self.screen_height = self.root.winfo_screenheight()
            self.window_width = self.screen_width // 2
            self.window_height = self.screen_height // 2
            self.x_offset = (self.window_width - (self.columns * self.field_size)) // 2
            self.y_offset = (self.window_height - (self.rows * self.field_size)) // 2
            self.main_frame = tk.Frame(self.root)
            self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.top_frame = tk.Frame(self.root)
            self.top_frame.pack(side=tk.TOP, fill=tk.X)

            # Move canvas into main_frame
            self.canvas = tk.Canvas(self.main_frame, width=self.window_width, height=self.window_height)
        self.current_mode = tk.StringVar(value="Play Mode")
        self.top_frame.grid_columnconfigure(0, weight=1)  # This centers mode_la
        self.canvas.pack()
        self.board_config_list = None
        self.numbers_list = None
        self.board_array = np.zeros((self.rows, self.columns), dtype=int)
        self.draw_board()

        self.canvas.bind("<Button-1>", self.click_event)

        self.starting_player_var = tk.StringVar(value="1")
        self.start_button_red = tk.Radiobutton(root, text="Start as Red", variable=self.starting_player_var, value="1")
        self.start_button_red.pack(side=tk.LEFT)

        self.start_button_yellow = tk.Radiobutton(root, text="Start as Yellow", variable=self.starting_player_var, value="2")
        self.start_button_yellow.pack(side=tk.LEFT)

        self.apply_button = tk.Button(root, text="Apply", command=self.set_starting_player)
        self.apply_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.root, text="Save Board", command=self.save_canvas)
        self.save_button.pack(side=tk.LEFT)
        self.filename_var = tk.StringVar(value="board.png")
        self.filename_entry = tk.Entry(self.root, textvariable=self.filename_var)
        self.filename_entry.pack(side=tk.LEFT)


        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT)
        
        # Initialize button with mode toggle functionality
        self.play_button = tk.Button(root, text="Switch to Load Mode", command=self.toggle_play_simulation)
        self.play_button.place(relx=0.8, rely=0)

        # Initialize label to display the current mode and status
        self.mode_label = tk.Label(self.main_frame, textvariable=self.current_mode)
        self.mode_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # display current player 
        self.current_player_label = tk.Label(self.main_frame, text=f"Current Player: {self.current_player}")
        self.current_player_label.place(relx=0.43, rely=0.43, anchor=tk.CENTER)


        self.board_config = None
        self.numbers = None
        # Initialize the slider
        self.slider = tk.Scale(self.root, from_=0, to=10, orient=tk.HORIZONTAL , command=self.update_board_state)
        self.slider.pack(side=tk.BOTTOM, fill=tk.X)
        # self.slider.bind("<Motion>", self.update_board_state)
        self.update_label()
    
    
    def update_board_state(self, event):
        print("update board state")
        idx = self.slider.get()
        if self.board_config_list is not None and self.numbers_list is not None:
            print(f"lenth of board config list {len(self.board_config_list)}")
            print(f"lenth of numbers list {len(self.numbers_list)}")
            print(f"Updating board at {idx}")
            self.load_board(self.board_config_list[int(idx)], self.numbers_list[int(idx)])
            self.highlight_max_number_row(self.numbers_list[int(idx)])  # Highlight the row with max numbe

    def toggle_play_simulation(self):
        next_mode = "Eval Mode" if self.current_mode.get() == "Play Mode" else "Play Mode"
        self.current_mode.set(next_mode)
        next_mode_text = f"Switch to {'Play Mode' if next_mode == 'Eval Mode' else 'Eval Mode'}"
        self.play_button.config(text=next_mode_text)
        self.update_label()
        print(f"toggling mode to {next_mode}")
        # clear board
        self.reset_board()
              
    
    def update_label(self):
        print(" current player update label")
        # new_label = f"Mode: {self.current_mode.get()}, Status: {self.play_simulation_mode.get()}"
        color = self.player_colors[self.current_player]
        self.current_player_label.config(text=f"Current Player: {self.current_player}", fg=color)
        
        
        

    def reset_board(self):
        self.board_array = np.zeros((self.rows, self.columns), dtype=int)
        self.draw_board()

    def set_starting_player(self):
        self.current_player = int(self.starting_player_var.get())
        # set color of current player
        if self.current_player == 1:
            self.player_colors = {1: 'red', 2: 'yellow'}
        else:
            self.player_colors = {1: 'yellow', 2: 'red'}
        self.board_array = np.zeros((self.rows, self.columns), dtype=int)
        self.draw_board()

    def click_event(self, event):
        print(f"click event {event} current player {self.current_player}" )
        if self.current_mode.get() == "Play Mode":
            if self.blinking:
                self.reset_board()
            self.blinking = False
            column_clicked = (event.x - self.x_offset) // self.field_size
            if 0 <= column_clicked < self.columns:
                if self.update_board(column_clicked):
                    self.draw_board()
                    if self.check_winner():
                        print(f"Player {self.current_player} wins!")
                        self.highlight_winner()   
                    else:
                        self.current_player = 3 - self.current_player
            self.update_label()
        elif self.current_mode.get() == "Eval Mode":
            self.reset_board()
            # Example board config and numbers to show; you can read this from a file
            self.display_data()
    
    def display_data(self, board_config=None, numbers=None):
        if board_config is None:
            board_config = self.board_config_list
        if numbers is None:
            numbers = self.numbers_list
        self.reset_board()
        idx = self.slider.get()
        idx = 20
        print(f" runs simulation Slicing at {idx}")
        self.load_board(board_config[idx], numbers[idx])
      
        
    
    def highlight_max_number_row(self, numbers):
        max_number_column = np.argmax(numbers)  # Find the column with the max number
        free_row = np.argmax(self.board_array[:, max_number_column] > 0)  # Find the topmost filled cell in that column

        if free_row == 0:
            free_row = self.rows  # If column is empty, point to the bottom cell
        free_row -= 1  # Adjust to 0-based index

        # Highlight only the first row in the column
        x1 = max_number_column * self.field_size + self.x_offset
        y1 = free_row * self.field_size + self.y_offset
        x2 = x1 + self.field_size
        y2 = y1 + self.field_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='green', stipple='gray50', outline='black')

        # Redraw the number over the rectangle
        x = max_number_column * self.field_size + self.x_offset + self.field_size // 2
        y = free_row * self.field_size + self.y_offset + self.field_size // 2
        self.canvas.create_text(x, y, text=str(numbers[max_number_column]), font=("Arial", 10))


    def highlight_winner(self):
        assert self.winning_coords, "No winning coordinates found!"
        self.logger.info(f"Winning coordinates: {self.winning_coords}")
        
        self.blink_color_win = 'yellow' if self.current_player == 2 else 'red'
        self.blinking = True
        self.toggle_highlight()
    
    def toggle_highlight(self):
        if not self.blinking:
            return
        color = self.blink_color_win if self.blink_color == 'white' else 'white'
        self.blink_color = color

        for row, col in self.winning_coords:
            x1 = col * self.field_size + self.x_offset
            y1 = row * self.field_size + self.y_offset
            x2 = x1 + self.field_size
            y2 = y1 + self.field_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
        self.root.after(500, lambda: self.toggle_highlight())

    def update_board(self, column):
        for i in reversed(range(self.rows)):
            if self.board_array[i, column] == 0:
                self.board_array[i, column] = self.current_player
                return True
        return False

    def check_winner(self):
        self.winning_coords = []
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if all(self.board_array[row, col:col + 4] == self.current_player):
                    self.winning_coords = [(row, col + i) for i in range(4)]
                    return True

        for col in range(self.columns):
            for row in range(self.rows - 3):
                if all(self.board_array[row:row + 4, col] == self.current_player):
                    self.winning_coords = [(row + i, col) for i in range(4)]
                    return True

        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if all([self.board_array[row + i, col + i] == self.current_player for i in range(4)]):
                    self.winning_coords = [(row + i, col + i) for i in range(4)]
                    return True

        for row in range(3, self.rows):
            for col in range(self.columns - 3):
                if all([self.board_array[row - i, col + i] == self.current_player for i in range(4)]):
                    self.winning_coords = [(row - i, col + i) for i in range(4)]
                    return True

        return False

    def draw_board(self, numbers_to_display=None):

        for i in range(self.rows):
            for j in range(self.columns):
                x1 = j * self.field_size + self.x_offset
                y1 = i * self.field_size + self.y_offset
                x2 = x1 + self.field_size
                y2 = y1 + self.field_size
                color = 'white'
                if self.board_array[i, j] == 1:
                    color = 'red'
                elif self.board_array[i, j] == 2:
                    color = 'yellow'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
        if numbers_to_display is not None: 
            self.draw_numbers(numbers_to_display)
        


    def save_canvas(self):
        filename = self.filename_var.get()
        if not filename.endswith(".png"):
            filename += ".png"
        self.canvas.postscript(file="temp_canvas.ps")
        img = Image.open("temp_canvas.ps")
        img.save(filename, "png")
    
    def draw_numbers(self, numbers):
        for col in range(self.columns):
            free_row = np.argmax(self.board_array[:, col] > 0)
            #import pdb; pdb.set_trace()
            #print(f"free row {free_row}")
            if free_row == 0:
                free_row = self.rows
            free_row -= 1  # Move to the first free row from the bottom
            x = col * self.field_size + self.x_offset + self.field_size // 2
            y = free_row * self.field_size + self.y_offset + self.field_size // 2
            # print(f"draw numebers len {len(numbers)}" )
            try:
                self.canvas.create_text(x, y, text=str(numbers[col]), font=("Arial", 10))
            except:
                print(f"draw numebers len {len(numbers)}")
                print(f"col {col}")
                # import pdb; pdb.set_trace()

    def load_board(self, board_config, numbers):
        self.reset_board()
        self.board_array = np.array(board_config)
        self.draw_board()
        self.draw_numbers(numbers)
    

    def run_simulation(self, board_config_list=None, numbers_list=None, path="images/"):
        # create random board config
        self.slider.config(from_=0, to=len(board_config_list) - 1)
        os.makedirs(path, exist_ok=True)
        for idx, board_config, numbers in enumerate(zip(board_config_list, numbers_list)):
            print(f"Generating board {idx}")
            self.logger.info(f"Generating board {idx}")
            filename = os.path.join(path, f"board_{idx}.png")
            # board_config = np.random.randint(0, 3, size=(self.rows, self.columns))
            # numbers = np.random.randint(0, 100, size=self.columns)
            self.load_board(board_config=board_config, numbers=numbers)
            self.filename_var.set(f"{filename}")
            self.save_canvas()
            self.root.after(1000, self.update_label)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    root = tk.Tk()
    game = ConnectFour(root, logger)
    root.mainloop()