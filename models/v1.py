import torch
from torch import nn
import numpy as np 
import game_functions as gf

class PolicyNetwork(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten() 
        # Neural network architecture:
        self.linear_relu_softmax = nn.Sequential(
            nn.Linear(2*(4*4) , 10),
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
        self.history = []

    def observation(self):
        if len(self.history)>=2:
            return np.stack((self.history[-1], self.history[-2]))
        else:
            return np.stack((self.history[-1], np.zeros_like(self.history[-1])))
    
    def agent_key_press(self):
        # Obtain observation: 
        state = self.observation()
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
            arr_new = shift_func(self.history[-1])
            if (arr_new == self.history[-1]).all():
                probs[n] = 0

        # Renormalize probabilities: 
        probs = probs/np.sum(probs)
        action_choice = np.random.choice(action_indices , p = probs)
        
        # Append logit value to the agent's logits attribute: 
        self.logits.append(logit[0][action_choice])
        return action_dict[action_choice][0]