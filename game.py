##########################################################################################################################################################
# This is a version of the game of human control.
##########################################################################################################################################################
import numpy as np
from itertools import groupby, chain
import curses 


def sum_elements_left(arr): 
    groupby_array =[list(g) for k , g in groupby(arr)]
    new = []
    for g in groupby_array: 
        if len(g)==3 and g[0] != 0: 
            new.append([g[0]+g[1]] + g[2:])
        elif len(g)==4 and g[0] != 0: 
            new.append([g[0] + g[1]] + [g[2] + g[3]])
        elif len(g) == 2 and g[0]!=0: 
            new.append([sum(g)])
        else: 
            new.append(g)
    new_chained = np.array(list(chain(*new)))
    return new_chained

def sum_elements_right(arr): 
    arr = np.flip(arr)
    groupby_array =[list(g) for k , g in groupby(arr)]
    new = []
    for g in groupby_array: 
        if len(g)==3 and g[0] != 0: 
            new.append([g[0]+g[1]] + g[2:])
        elif len(g)==4 and g[0] != 0: 
            new.append([g[0] + g[1]] + [g[2] + g[3]])
        elif len(g) == 2 and g[0]!=0: 
            new.append([sum(g)])
        else: 
            new.append(g)
    new_chained = np.flip(np.array(list(chain(*new))))
    return new_chained

def new_tile(arr): 
    zero_indices = np.where(arr == 0 )
    insert_two = (np.random.uniform() <= 0.9)
    position_of_new_tile = np.random.choice(len(zero_indices[0]))
    new_tile_index = (zero_indices[0][position_of_new_tile] , zero_indices[1][position_of_new_tile])
    arr_new = arr.copy() 
    if insert_two == True: 
        arr_new[new_tile_index] += 2 
    else: 
        arr_new[new_tile_index] += 4
    return arr_new

class GameEnv: 
    '''Environment for game'''
    def __init__(self , game_state): 
        self.game_state = game_state

    def push_left(self):
        arr = self.game_state
        shifted_arrays = []
        for i in range(arr.shape[0]):
            arr_slice = arr[i]
            non_zero = arr_slice[arr_slice!=0]
            summed = sum_elements_left(non_zero)
            summed_pad = np.concatenate((summed , np.zeros(4 - len(summed))))
            shifted_arrays.append(summed_pad)
        shifted_arrays = np.array(shifted_arrays, dtype = np.int64)
        if (shifted_arrays == arr).all():
            self.game_state = shifted_arrays
        else: 
            shifted_arrays_new = new_tile(shifted_arrays)
            self.game_state = shifted_arrays_new

    def push_right(self): 
        arr = self.game_state
        shifted_arrays = []
        for i in range(arr.shape[0]):
            arr_slice = arr[i]
            non_zero = arr_slice[arr_slice!=0]
            summed = sum_elements_right(non_zero)
            summed_pad = np.concatenate((np.zeros(4 - len(summed)), summed ))
            shifted_arrays.append(summed_pad)
        shifted_arrays = np.array(shifted_arrays, dtype = np.int64)
        if (shifted_arrays == arr).all():
            self.game_state = shifted_arrays
        else: 
            shifted_arrays_new = new_tile(shifted_arrays)
            self.game_state = shifted_arrays_new

    def push_up(self): 
        arr = self.game_state
        shifted_arrays = []
        for i in range(arr.shape[0]):
            arr_slice = arr[: , i]
            non_zero = arr_slice[arr_slice!=0]
            summed = sum_elements_left(non_zero)
            summed_pad = np.concatenate((summed, np.zeros(4 - len(summed)) ))
            shifted_arrays.append(summed_pad)
        shifted_arrays =  np.array(shifted_arrays, dtype = np.int64).T
        if (shifted_arrays == arr).all():
            self.game_state = shifted_arrays
        else: 
            shifted_arrays_new = new_tile(shifted_arrays)
            self.game_state = shifted_arrays_new

    def push_down(self):
        arr = self.game_state
        shifted_arrays = []
        for i in range(arr.shape[0]):
            arr_slice = arr[: , i]
            non_zero = arr_slice[arr_slice!=0]
            summed = sum_elements_right(non_zero)
            summed_pad = np.concatenate((np.zeros(4 - len(summed)), summed  ))
            shifted_arrays.append(summed_pad)
        shifted_arrays = np.array(shifted_arrays, dtype = np.int64).T
        if (shifted_arrays == arr).all():
            self.game_state = shifted_arrays
        else: 
            shifted_arrays_new = new_tile(shifted_arrays)
            self.game_state = shifted_arrays_new

    def render(self): 
        return str(self.game_state)

class InputHandler: 
    def __init__(self , game: GameEnv): 
        self.game = game 
        self.key_map = {curses.KEY_UP: self.game.push_up,
            curses.KEY_DOWN: self.game.push_down,
            curses.KEY_LEFT: self.game.push_left,
            curses.KEY_RIGHT: self.game.push_right,}

    def handle(self, key):
        """Returns False to signal quit, True otherwise."""
        if key in (ord('q'), 27):  # 'q' or plain Esc to quit
            return False
        action = self.key_map.get(key)
        if action:
            action()
        return True

class GameLoop:
    def __init__(self, game: GameEnv):
        self.game = game
        self.input_handler = InputHandler(game)

    def run(self, stdscr):
        curses.curs_set(0)      # hide the cursor
        stdscr.keypad(True)     # decode arrow-key escape sequences for us
        running = True

        while running:
            stdscr.clear()
            stdscr.addstr(0, 0, self.game.render())
            stdscr.refresh()

            key = stdscr.getch()  # blocks until a key is pressed
            running = self.input_handler.handle(key)


def main():
    game = GameEnv(np.array([[0 , 0 , 0 , 0] , [0 , 2 , 0 , 0] , [0 , 0 , 2 , 0 ] , [0 , 0 ,0 , 0]]))
    loop = GameLoop(game)
    curses.wrapper(loop.run)


if __name__ == "__main__":
    main()
