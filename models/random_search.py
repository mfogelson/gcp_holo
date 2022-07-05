import numpy as np 
from timebudget import timebudget

# @timebudget
def random_search(env, episodes=100):
    """random search for linkage graph generation

    Args:
        env (gym.env): linkag_gym
        episodes (int, optional): number of linkage graphs to generate. Defaults to 100.

    Returns:
        (dict, list, list, list): Best designs from search, all rewards, all designs, all episode lengths
    """
    rewards = []
    designs = []
    lengths = []
    
    ## run for episodes
    for _ in range(episodes):
        env.reset()
        done = False
        l = 0
        while not done:
           obs, reward, done, info = env.apply_random_action()
           l += 1
        
        ## Update lists
        rewards.append(reward)
        designs.append(info)
        lengths.append(l)
        
            
    return env.best_designs, rewards, designs, lengths

