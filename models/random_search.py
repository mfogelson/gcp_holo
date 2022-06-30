import numpy as np 
from timebudget import timebudget

# @timebudget
def random_search(env, timesteps=100):
    rewards = []
    designs = []
    lengths = []
    for _ in range(timesteps):
        env.reset()
        done = False
        l = 0
        while not done:
           obs, reward, done, info = env.apply_random_action()
           l += 1
        rewards.append(reward)
        designs.append(info)
        lengths.append(l)
        
            
    return env.best_designs, rewards, designs, lengths

# @timebudget
def random_search_full(env, timesteps=100):
    rewards = []
    designs = []
    lengths = []
    for _ in range(timesteps):
        env.reset()
        done = False
        l = 0
        while not done:
           obs, reward, done, info = env.apply_random_action()
           env._initialize_paths()
           l += 1
        rewards.append(reward)
        designs.append(info)
        lengths.append(l)
        
            
    return env.best_designs, rewards, designs, lengths

# def random_lipson_search(env, timesteps=100, heuristic=False):
#     rewards = []
#     designs = []
#     lengths = []

#     if env.is_valid():
#         for _ in range(timesteps):
#             while not done:
#                 state = env.get_observation()

#                 # Select pair of nodes
#                 action1 = action_RS.choice(np.arange(env.get_edges().shape[0]-2)) #combinations(range(state.number_of_nodes()), 2) 
#                 action1 = env.get_edges()[action1+2,:]
#                 # Get valid positions
                

#                 if action_RS.rand() > 0.5:
#                     # Action T
#                     # Select random valid edge position
#                     if heuristic:
#                         edge_mask = env.get_edge_mask(action1[0], action1[1])
#                         if sum(edge_mask) == 0 :
#                             return replay_buffer, None
#                         action2 = action_RS.choice(pos_ind, size=1, replace=False, p=edge_mask/sum(edge_mask))#edge_mask.choice()
#                         action2 = env.grid[action2,:][0]
#                     else:
#                         action2 = action_RS.rand(2)

#                     env.step_T([action1, action2])
#                     action = [np.array([0]), action1, action2]
#                 else:
#                     # Action D
#                     valid_nodes = list(range(env.number_of_nodes()))
#                     valid_nodes.remove(action1[0])
#                     valid_nodes.remove(action1[1])
#                     action2a = action_RS.choice(valid_nodes)
#                     action2b = action_RS.rand(2)
#                     env.step_D([action1, action2a, action2b])
#                     action = [np.array([1]), action1, action2a, action2b]
            
            
#     return env.best_designs, rewards, designs, lengths

