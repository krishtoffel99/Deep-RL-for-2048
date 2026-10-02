from game_agent_friendly import PolicyNetwork, Agent, generate
import torch
import numpy as np
import torch.optim as optim

# Get GPU device if available:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Initialise policy network
policy_model = PolicyNetwork().to(device)

# Initialise start state of the game:
state_0 = np.array([[0 , 0 , 0 , 0] , [0 , 2 , 0 , 0] , [0 , 0 , 2 , 0 ] , [0 , 0 ,0 , 0]])

# Helper function for computing future returns from time step t:
def G_t(t, rewards, gamma): 
    '''
    Computes return in state at time t
    '''
    d = np.array([gamma**i for i in range(len(rewards) - t)])
    r = rewards[t:]
    return d@r

# Helper function for computing the loss associated with an episode: 
def reinforce_loss(agent, rewards, gamma):
    '''
    Computes the scalar loss function for the REINFORCE policy gradient.
    Inputs: 
    agent: Agent class type, 
    rewards: list of rewards per time step over episode. 
    gamma: float,  discount factor (between 0 and 1)
    Returns: 
    loss: scalar loss.
    '''
    # Compute list of returns:
    G = [G_t(t , rewards, gamma) for t in range(len(rewards))]
    return_tensor = torch.tensor(G, dtype = torch.float32 , device = device, requires_grad = False)
    return_tensor = (return_tensor - return_tensor.mean())/(return_tensor.std() +1e-8)
    # Extract logits from episode: 
    logits_tensor = torch.stack(agent.logits)

    # Scalar loss function: 
    loss = -return_tensor.dot(logits_tensor)
    return G, loss 

####################################################################################################
# TRAINING 
####################################################################################################
NUM_EPISODES = 5000
LEARNING_RATE = 1e-6
GAMMA = 0.9
# Using an ADAM optimiser for an initial test
optimiser = optim.Adam(policy_model.parameters(), lr = LEARNING_RATE, weight_decay=1e-5)

# Initialising tensorboard writer: 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir = "runs/vanilla_REINFORCE")

def compute_grad_norm(model):
    total_norm_sq = 0.0 
    for p in model.parameters(): 
        if p.grad is not None: 
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item()**2
    return total_norm_sq**0.5

for n in range(NUM_EPISODES):
    print('RUNNING EPISODE: ', n)
    optimiser.zero_grad()
    # Initialise new agent: 
    agent = Agent(policy_model, [], [], device)
    # Generate an episode to train: 
    states, actions, rewards = generate(state_0 , agent)

    # Compute loss: 
    G, loss = reinforce_loss(agent, rewards, GAMMA)
    # Compute gradients: 
    loss.backward() 
    # Logging stuff: 
    torch.nn.utils.clip_grad_norm_(policy_model.parameters() , max_norm = 5.0)
    writer.add_scalar("reward/episode_maxtile" , np.max(states[-1]), n)
    writer.add_scalar("reward/episode_initial_return" , G[0], n)
    writer.add_scalar("reward/episode_rewards" , sum(rewards) , n)
    writer.add_scalar("episode/episode_lengths" , len(states) , n)
    writer.add_scalar("diagnostics/grad_norm" , compute_grad_norm(policy_model) , n)
    writer.add_scalar("diagnostics/policy_entropy" , np.mean(agent.entropy), n)
    # Implement optimisation step: 
    optimiser.step()

writer.close()


