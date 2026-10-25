##########################################################################################################################################################
# This version of the game lets a deep RL agent take control. Used for generating episodes of experience to learn from.
##########################################################################################################################################################
import numpy as np
from itertools import groupby, chain
import torch
import game_functions as gf
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
    # Compute zero tile locations:
    zero_indices = np.where(arr == 0)
    # Probabilistically include a 2 tile:
    insert_two = (np.random.uniform() <= 0.9)
    # Probabilistically compute location of new tile:
    position_of_new_tile = np.random.choice(len(zero_indices[0]))
    # Compute new tile index:
    new_tile_index = (zero_indices[0][position_of_new_tile] , zero_indices[1][position_of_new_tile])
    # Insert tiles:
    arr_new = arr.copy() 
    if insert_two == True: 
        arr_new[new_tile_index] += 2 
    else: 
        arr_new[new_tile_index] += 4
    return arr_new

class GameEnv: 
    '''Environment for game'''
    def __init__(self , game_state, game_seed = None): 
        self.game_state = game_state
        self.game_seed = game_seed

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

    def log_state(self): 
        return self.game_state

    def gameover_check(self): 
        state = self.game_state
        # Check each row for consecutive duplicates: 
        for i in range(state.shape[0]): 
            vec = state[i , :]
            # Define a boolean which checks if *any* consecutive elements of row are equal
            can_move =(np.array([len(list(g))>=2 for k, g in groupby(vec)])==True).any()
            if can_move: 
                # Return False as game is not over:
                return False

        # Check each row for consecutive duplicates: 
        for i in range(state.shape[0]): 
            vec = state[: , i]
            # Define a boolean which checks if *any* consecutive elements of row are equal
            can_move =(np.array([len(list(g))>=2 for k, g in groupby(vec)])==True).any()
            if can_move: 
                # Return False as game is not over:
                return False


class InputHandler: 
    def __init__(self , game: GameEnv): 
        self.game = game 
        self.key_map = {'up': self.game.push_up,
            'down': self.game.push_down,
            'left': self.game.push_left,
            'right': self.game.push_right,}

    def handle(self, key):
        """Returns False to signal quit, True otherwise."""
        if key == 'quit':  # Agent can hit the quit key in which case handle returns 'False'. This will cause the episode to terminate in GameLoop.
            return False
        action = self.key_map.get(key)
        if action:
            action()
        return True # If the agent does not hit 'quit', handle calls the function in key_map, thus shifting the agent to a new state.
    
from torch import nn
class PolicyNetwork(nn.Module): 
    def __init__(self): 
        super().__init__() # Not sure what this is but it looks like it's standard to stick at the start of the class.
        self.flatten = nn.Flatten() # this flattens the data array which might not be a vector but rather a matrix.
        # Neural network architecture:
        self.linear_relu_softmax = nn.Sequential(
            nn.Linear(4*4 , 10),
            nn.ReLU(), 
            nn.Linear(10 , 6), 
            nn.ReLU(), 
            nn.Linear(6 , 4), 
            nn.LogSoftmax(dim = 1)
        )

    # Define a forward pass through the network
    def forward(self , x): 
        # Invoke flatten to convert data array into a vector:
        x = self.flatten(x)
        # Run the input vector through the neural network to compute the outputs (called 'logits')
        logits = self.linear_relu_softmax(x)
        return logits

class Agent: 
    def __init__(self , policy: PolicyNetwork, logits, entropy, device):
        self.policy = policy
        self.device = device
        self.logits = logits
        self.entropy = entropy
    def agent_key_press(self, state):
        # Load state as a normalised vector: 
        mask = (state > 0)
        state_normed = np.where(mask , np.log2(state), state)/11
        # Load game state into a torch tensor:
        torch_game_state = torch.tensor(np.array([state_normed]), dtype = torch.float32, device=self.device)

        # Compute agent's logits: 
        logit = self.policy(torch_game_state)
        # Extract probabilities associated with the prediction: 
        probs = torch.exp(logit[0]).detach().numpy()
        # Computing policy entropy for diagnosis: 
        self.entropy.append(-probs.dot(np.log(probs)))
        # Sample distribution to obtain action choice
        action_indices = [0 , 1 , 2 , 3 ] 
        # Look over possible actions and determine illegal moves. Set these probabilities to zero to prevent the agent from trying futile moves
        action_dict = {0 : ('up', gf.push_up) , 1 : ('down', gf.push_down) , 2 : ('left',gf.push_left) , 3 : ('right', gf.push_right)}
        for n in action_indices: 
            shift_func = action_dict[n][1]
            arr_new = shift_func(state)
            if (arr_new == state).all():
                probs[n] = 0
        print(probs)
        # Renormalize probabilities: 
        probs = probs/np.sum(probs)
        action_choice = np.random.choice(action_indices , p = probs)
        
        # Append logit value to the agent's logits attribute: 
        self.logits.append(logit[0][action_choice])
        return action_dict[action_choice][0]

def count_pairs(state): 
    num = 0
    for i in range(state.shape[0]):
        vec = state[i , :]
        groups = [list(g) for k,g in groupby(vec)]
        num += sum([(len(g)>1) for g in groups])
    for j in range(state.shape[1]):
        vec = state[: , j]
        groups = [list(g) for k,g in groupby(vec)]
        num += sum([(len(g)>1) for g in groups])
    return num

class GameLoop:
    def __init__(self, game: GameEnv, agent: Agent):
        self.game = game
        self.agent = agent
        self.input_handler = InputHandler(game)


    def run(self, print_episode):
        states = []
        # Append starting state: 
        states.append(self.game.game_state)
        # List to store actions
        actions = []
        # List to store rewards: 
        rewards = []
        # Allow the agent to play
        running = True
        while running:
            # Generate action through policy
            action = self.agent.agent_key_press(self.game.game_state)
            # Input action
            running = self.input_handler.handle(action)
            # Append action:
            actions.append(action)

            #REWARD########################################################################################
            # Compute reward associated with action: 
            # Provide a reward that is equal 
            # Provide a reward if the agent increases the maximum tile on the board:
            if (np.max(self.game.game_state)==np.max(states[-1])): 
                r = 0
            else: 
                r = np.max(self.game.game_state) - np.max(states[-1])

            # # Give an extra reward proportional to the difference in number of neighbouring tiles:
            r += (count_pairs(self.game.game_state) - count_pairs(states[-1]))
            # Supply negative rewards for taking an action that does not change the game-board:
            if (self.game.game_state == states[-1]).all(): 
                r += -2

            # Supplying further negative rewards and terminating the state if the agent visits the same state 4 times in a row: 
            # if (len(states) >= 4) and (self.game.game_state == states[-3]).all(): 
            #     r = -5 

            rewards.append(r)

            if (len(states) >= 4) and (self.game.game_state == states[-3]).all(): 
                running = False
            #END REWARD###################################################################################

            # Check if agent trapped in terminal state (game over):
            gameover = self.game.gameover_check()

            if (gameover == None) and (len(np.argwhere(self.game.game_state == 0))==0):
                # If all cells are filled and no scope for further moves, terminate the game loop:
                states.append(self.game.game_state)
                running = False 
            else: 
                states.append(self.game.game_state)
            

            if print_episode:
                print(action)
                print(self.game.game_state)
            
        return states, actions, rewards


def generate(state_0, agent, seed = None,  print_episode = False):
    game = GameEnv(state_0 , seed)
    # Initialise game seed if supplied: 
    np.random.seed(game.game_seed)
    # Run game loop to generate an episode:
    loop = GameLoop(game, agent)
    episode = loop.run(print_episode)
    return episode


