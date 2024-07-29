from kaggle_environments import make, evaluate
import torch
import numpy as np
import random

def get_outcomes(agent1, agent2, n_rounds=100):
    config = {'rows': 6, 'columns': 7, 'inarow': 4}       
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)   
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    
    wins  = outcomes.count([1,-1]) + outcomes.count([0,None]) 
    loses = outcomes.count([-1,1]) + outcomes.count([None,0]) 
    draws = n_rounds - wins - loses
    
    return wins, draws, loses 


def print_outcomes(outcomes):
    print(f"Wins: {outcomes[0]}, Draws: {outcomes[1]}, Loses: {outcomes[2]}")

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)