# from env.Mech import Mech
import numpy as np 


def get_valid_env():
    # Grid
    m = 11 
    x = np.linspace(0,1,num=m)
    i, j = np.meshgrid(x, x)
    i = i.flatten()
    j = j.flatten()
    pos = np.vstack([i, j]).T
    pos_ind = range(pos.shape[0])

    initial_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


    node_pos_ind = np.random.choice(pos_ind, size=4, replace=False)
    node_pos = pos[node_pos_ind,:]
    env = Mech()
    env.init(node_positions=node_pos, edges=initial_edges, steps=50)
    # Step 2 check validity 
    while not env.is_valid():
        # If not valid save data
        node_pos_ind = np.random.choice(pos_ind, size=4, replace=False)
        node_pos = pos[node_pos_ind,:]
        env = Mech()
        env.init(node_positions=node_pos, edges=initial_edges, steps=50)

    return env