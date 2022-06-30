import numpy as np 
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch 
from torch_geometric.data import Data
from dtaidistance import dtw_ndim
import scipy
from model.rollout_buffer import RolloutBuffer
from copy import deepcopy
import pdb
import cma

# from jax.numpy.linalg import svd
def state_to_torch(state):
    '''
    Converts gym observation to torch data
    :params
    state -> dictionary (keys [adj, x, mask])

    :returns
    torch_geometric.Data()
    '''
    adj = torch.from_numpy(state['adj']).float()
    x = torch.from_numpy(state['x']).float()
    mask = torch.from_numpy(state['mask']).bool()
    node_pos = torch.from_numpy(state['node_pos']).float()
    data = Data(adj=adj, x=x, mask=mask, node_pos=node_pos)

    return data

def normalize_curve(input, scale=None, shift=None):
    '''
    :params input-> 2xN
    :return output -> 2xN
    '''
    if input.shape[0] != 2:
        input = input.T
        
    N = input.shape[1]
    mu = np.mean(input, axis=1).reshape(-1,1)
    med = np.median(input, axis=1).reshape(-1,1)
    std = max(np.std(input, axis=1).reshape(-1,1))

    # # pdb.set_trace()
    # Cxx = 1./N*np.sum((input-mu)**2, axis=1)
    # Cxy = 1./N*np.sum((input[0,:]-mu[0,0])*(input[1,:]-mu[1,0]))

    # C = np.array([[Cxx[0], Cxy], [Cxy, Cxx[1]]])

    # _, v = np.linalg.eig(C)

    output = (input - mu) #- med) # mu) # Shift
    # output = np.matmul(v.T, output) # Rotate
    output /= (std+1e-10) # Scale
    
    if scale is not None:
        output*=scale
    
    if shift is not None: 
        output+=shift

    return output

def new_dist(goal, coupler):
    if goal.shape[1] != 2:
        goal = goal.T
        
    if coupler.shape[1] != 2:
        coupler = coupler.T 
    
    min_dist = 1000
    min_R = None
    min_i = None
    for i in np.arange(coupler.shape[0]):
        tmp = np.roll(coupler, 2*(i))  
        H = goal.T@tmp
        U, s, Vh = scipy.linalg.svd(H, full_matrices=False)
        # U, s, Vh = svd(H, full_matrices=False)

        R = Vh@U
        tmp = tmp @ R
        # R = None
        dist = np.linalg.norm(tmp-goal)
        
        if dist < min_dist:
            min_R = R
            min_dist = dist
            min_i = 2*(i)
            dir = 1
            
        tmp = tmp[::-1]
        
        H = goal.T@tmp
        U, s, Vh = scipy.linalg.svd(H, full_matrices=False)
        # U, s, Vh = svd(H, full_matrices=False)

        R = Vh@U
        tmp = tmp @ R
        # R = None
        dist = np.linalg.norm(tmp-goal)
        
        if dist < min_dist:
            min_R = R
            min_dist = dist
            min_i = 2*(i)
            dir = -1
            
    return min_dist, min_R, min_i, dir 
        
def distance(goal, coupler, ordered=False, distance_metric='euclidean'):
    '''
    NOTE: This must be Nx2 
    '''
    if goal.shape[1] != 2:
        goal = goal.T
        
    if coupler.shape[1] != 2:
        coupler = coupler.T 
        
    C = cdist(goal, coupler, metric=distance_metric)
    
    row_ind = np.arange(goal.shape[0])
    
    if ordered:
        row_inds = row_ind[row_ind[:,None]-np.zeros_like(row_ind)].T
        col_inds = row_ind[row_ind[:,None]-row_ind].T 
        
        min_clock_wise = np.amin(C[row_inds, col_inds].sum(1))
        argmin_clock_wise = np.argmin(C[row_inds, col_inds].sum(1))
        
        min_count_clock_wise = np.amin(C[row_inds, col_inds[:,::-1]].sum(1))
        argmin_count_clock_wise = np.argmin(C[row_inds, col_inds[:,::-1]].sum(1))
        
        cw_dir = int(min_clock_wise < min_count_clock_wise)
        # pdb.set_trace()
        col_ind = col_inds[int(cw_dir * argmin_clock_wise + (1-cw_dir)*argmin_count_clock_wise), :]
        # col_ind = col_inds[np.argmin(C[row_inds, col_inds].sum(1)), :]
    else:   
        row_ind, col_ind = linear_sum_assignment(C)
            
    return C[row_ind, col_ind]

def dtw_distance(goal, coupler):
    '''
    NOTE: This must be Nx2 
    '''
    if goal.shape[1] != 2:
        goal = goal.T
        
    if coupler.shape[1] != 2:
        coupler = coupler.T 
        
    dist = dtw_ndim.distance(goal, coupler)

    return dist

def uniquify(path):
    import os
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def buffer_to_env(buffer):
    '''
    :params\n
    state buffer -> Type: torch.Tensor\n

    :returns\n
    node_positions -> Type: Nd.Array | Shape: (N, 50, 2)\n
    edges -> Type: Nd.Array | Shape: (E, 2) \n
    goal -> Type: None or Nd.Array Shape: (2, 50)
    '''
    goal = None
    N, D = buffer.x.shape
    
    node_positions = buffer.node_pos.numpy()#[:,1:101].reshape(N,50,2, order='f') #final_designs[i].x.numpy()
    edges = np.argwhere(buffer.adj.numpy() > 0).T # NOTE: E x 2
    if D > 101:
        goal = buffer.x.numpy()[0,-100:].reshape(2,50)

    return node_positions, edges, goal

def dfs(visited, edges, graph, node, known_joints):
    
    if node not in visited and node not in [0, 1, 2]:
        visited.add(node)

        neighbours = np.array(known_joints)[np.where(graph.adj[int(node), known_joints] >= 1)[0]][:2]
        # print(node, neighbours)
        edges.add(frozenset((node, neighbours[0])))
        edges.add(frozenset((node, neighbours[1])))
        for neighbour in neighbours:
            dfs(visited, edges, graph, neighbour, known_joints)
            
def prune_buffer(visited, buffer, num_init_nodes):
    new_buffer = RolloutBuffer()
    idx = ((np.array(visited)-num_init_nodes)*2).astype(int)
    new_buffer.states.extend([buffer.states[i:i+2] for i in idx])
    new_buffer.actions.extend([buffer.actions[i:i+2] for i in idx])
    new_buffer.logprobs.extend([buffer.logprobs[i:i+2] for i in idx])
    new_buffer.rewards.extend([buffer.rewards[i:i+2] for i in idx])
    new_buffer.is_terminals.extend([buffer.is_terminals[i:i+2] for i in idx])
    
    return new_buffer

def order_env(env):
    known_joints = [0, 1, 2]
    n = env.number_of_nodes()
    unknown_joints = list(set(range(n)) ^ set(known_joints))
    order = {0:0, 1:1, 2:2}
    edges = []
    new_adj = np.zeros_like(env.adj)
    new_adj[0,1] = 1 
    new_adj[1,0] = 1
    new_adj[0,2] = 1  
    new_adj[2,0] = 1 
    new_paths = np.zeros_like(env.paths)
    new_paths[:3, :, :] = env.paths[:3, :, :]
    count = 3
    trys = 0
    max_trys = 500
    while list(set(range(n)) ^ set(known_joints)) != [] and trys < max_trys:
        # print(count)
            for i in unknown_joints[:]:
                
                if sum(env.adj[i, known_joints]) >= 2:

                    inds = np.array(known_joints)[np.where(env.adj[i, known_joints] >= 1)[0]]
                    new_paths[count,:,:] = env.paths[i, :,:]
                    
                    new_adj[count, order[inds[0]]] = 1
                    new_adj[order[inds[0]], count] = 1
                    
                    new_adj[count, order[inds[1]]] = 1
                    new_adj[order[inds[1]], count] = 1
                    
                    order[i] = count
                    
                    count+=1
                    unknown_joints.remove(i)
                    known_joints.append(i)
                else:
                    trys +=1
     
    if trys >= max_trys:
        return None, None, None, None   
        
    env_new = deepcopy(env)   
    edges = np.argwhere(new_adj == 1)
    env_new.init_args['node_positions'] = new_paths[:,:,0]
    env_new.init_args['edges'] = edges
    env_new.reset()

    return env_new, order, new_paths, new_adj

def prune_mech_new(env, order, idx):
    order = get_joint_order(env)
    remove_nodes = list(np.arange(env.max_nodes))
    remove_nodes.remove(0)
    remove_nodes.remove(1)
    remove_nodes.remove(2)
    for r in order[:idx+1]:
        remove_nodes.remove(r)
    
    env.paths = np.delete(env.paths, remove_nodes, 0)
    env.paths = np.append(env.paths, np.zeros([len(remove_nodes), 2, 50]), 0)
    env.adj = np.delete(env.adj, remove_nodes, 0)
    env.adj = np.delete(env.adj, remove_nodes, 1)
    env.adj = np.append(env.adj, np.zeros([len(remove_nodes), env.adj.shape[1]]), 0)
    env.adj = np.append(env.adj, np.zeros([env.adj.shape[0], len(remove_nodes)]), 1)
        
    n = env.number_of_nodes()
    env.init_args['node_positions'] = env.paths[:n,:,0]
    env.init_args['edges'] = env.get_edges()
    env.reset()
    
    
    return env
    

def prune_mech(env, idx):
    if np.any(np.abs(np.diff(env.paths[idx,:,:])) > 0):
        visited = set()
        edges = set()
        known_joints = np.arange(env.number_of_nodes())

        dfs(visited, edges, env, int(idx), known_joints)
        visited = list(visited)
        visited.sort()
        # new_buffer = None
        # if buffer is not None:
        #     new_buffer = prune_buffer(visited, buffer, env.number_of_initial_nodes())
        node_ids = [0, 1, 2] + visited
        edges = np.array([[0, 1], [0, 2]]+ [[node_ids.index(list(e)[0]), node_ids.index(list(e)[1])] for e in edges])
        # env.paths[:len(node_ids), :, :] = env.paths[node_ids,:,:]
        # env.paths[len(node_ids):, :, :] = 0.0
        # env.adj = np.zeros_like(env.adj)
        # env.adj[edges[0,:], edges[1,:]] = 1
        # env.adj[edges[1,:], edges[0,:]] = 1
        env.init_args['node_positions'] = env.paths[node_ids,:,0]
        env.init_args['edges'] = edges
        env.paths = np.zeros_like(env.paths)
        env.adj = np.zeros_like(env.adj)
        # node_positions = env.init_args['node_positions']
        # edges = env.init_args['edges']
        n = len(node_ids)

        env.paths[:n, :, 0] =  env.init_args['node_positions']

        env.adj[edges[:,0], edges[:,1]] = 1
        env.adj[edges[:,1], edges[:,0]] = 1

        env._initialize_paths()
        # pdb.set_trace()
        # env.reset()
        return env #, new_buffer
    return None

def update_best(env, best_rewards, best_env, init_n=3):
    sub_graphs = [prune_mech(deepcopy(env),idx) for idx in np.arange(init_n, env.max_nodes)]
    # sub_graphs = [e for e in sub_graphs if e.is_valid()]
    # # rewards = np.array([e._get_reward()[0] if e is not None and e.is_valid() else -100 for e in sub_graphs])

    # updates = rewards > best_rewards 
    # best_rewards[updates] = rewards[updates]
    # best_env = {k:(deepcopy(sub_graphs[k]) if update else v) for k, (update, v) in enumerate(zip(updates, best_env.values())) }
    for g in sub_graphs:
        if g is not None and g.is_valid(): 
            n = g.number_of_nodes()-init_n-1
            r = g._get_reward()[0]
            if r > best_env[n]._get_reward()[0]:
                best_env[n] = deepcopy(g)

    return best_env

def circIntersectionVect(jointA, jointB, jointC_pos):
    '''
    jointA = (2, N)
    jointB = (2, N)
    lengthC_pos = (2, )

    Return: 
    sol = (2, N)
    '''
    _, N = jointA.shape
    lengthA = np.linalg.norm(jointA[:,0] - jointC_pos)
    lengthB = np.linalg.norm(jointB[:,0] - jointC_pos)
    # pdb.set_trace()
    d = np.linalg.norm(jointB-jointA, axis=0).reshape(1,N) # (1, N)
    # assert d < lengthA + lengthB and d > abs(lengthA-lengthB)

    a = np.divide(lengthA**2 - lengthB**2 + np.power(d,2), 2.0*d)

    
    h = np.sqrt(lengthA**2-np.power(a,2))
    # if (np.isnan(h)).any():
    #     # pdb.set_trace()
    #     print("WARNING: Not Valid")

    P2 = jointA + np.divide(np.multiply(a, (jointB - jointA)),d)
    # print(P2.shape, h.shape, a.shape, d.shape)


    sol1x = P2[0,:] + np.divide(np.multiply(h, (jointB[1,:].reshape(1,N)- jointA[1,:].reshape(1,N))), d) 
    sol1y = P2[1,:] - np.divide(np.multiply(h, (jointB[0,:].reshape(1,N)- jointA[0,:].reshape(1,N))), d)
    # print(sol1x.shape)
    sol1 = np.vstack([sol1x, sol1y])
    # print(sol1.shape)

    sol2x = P2[0,:] - np.divide(np.multiply(h, (jointB[1,:].reshape(1,N)- jointA[1,:].reshape(1,N))), d) 
    sol2y = P2[1,:] + np.divide(np.multiply(h, (jointB[0,:].reshape(1,N)- jointA[0,:].reshape(1,N))), d)

    sol2 = np.vstack([sol2x, sol2y])
    
    if np.linalg.norm(sol1[:,0]-jointC_pos)<1e-4:
        return sol1
    elif np.linalg.norm(sol2[:,0]-jointC_pos)<1e-4:
        return sol2
    else:
        print(f"Error: Neither solution fit initial position, sol1 : {sol1[:,0]}, sol2: {sol2[:,0]}, orig: {jointC_pos} ")
        sol1[:] = np.nan
        return sol1
        # pdb.set_trace()

    # return sol1, sol2

def symbolic_kinematics(xi, xj, xk0):
    '''
    :params: \n
    xi = (2, N) known joint positions \n
    xj = (2, N) known joint positions 2 \n
    xk0 = (2, ) unkown joint initial position \n

    :return: \n
    xk = (2, N) joint positions for xk joint
    '''
    _, N = xi.shape
    l_ij = np.linalg.norm(xj-xi, axis=0).squeeze() # (N, )
    l_ik = np.linalg.norm(xi[:,0] - xk0) # float
    l_jk = np.linalg.norm(xj[:, 0] - xk0) # float

    # t = (l_ij**2 + (l_ik**2 - l_jk**2))/(2*l_ij*l_ik) # (N, )
    # if (abs(t) <= 1).all(): # Valid
    ## Triangle inequality ##
    valid = np.logical_and.reduce((np.all(l_ik+l_jk > l_ij), 
                                np.all(l_ik+l_ij > l_jk),  
                                np.all(l_jk+l_ij > l_ik), 
                                np.all(l_ij > 0), 
                                np.all(l_ik > 0)))  #all(l_ik+l_jk >= l_ij) and all(l_ij+l_ik >= l_jk) and all(l_ij+l_jk >= l_ik)
    
    if valid:
        f = l_ik / l_ij # (N, )

        t = (l_ij**2 + (l_ik**2 - l_jk**2))/(2*l_ij*l_ik) # (N, )
        R = np.array([[t, -np.sqrt(1.-t**2)], [np.sqrt(1.-t**2), t]]) # (2, 2, N)
        Q = (R*f).T # (N, 2, 2)

        diff = xj-xi # ()
        diff = diff[np.newaxis, :,:].T
        xk = np.matmul(Q,diff).T.squeeze() + xi
        if np.linalg.norm(xk[:,0]-xk0) < 1e-3:
            return xk
        else:
            sol1 = xk[:,0]
            R = np.array([[t, np.sqrt(1.-t**2)], [-np.sqrt(1.-t**2), t]]) # (2, 2, N)
            Q = (R*f).T # (N, 2, 2)
            xk = np.matmul(Q,diff).T.squeeze() + xi
            if np.linalg.norm(xk[:,0]-xk0) < 1e-3:
                return xk
            else: # Passes through singularity 
                sol2 = xk[:,0]
                # print(f"Error: Neither solution fit initial position, sol1 : {sol1}, sol2: {sol2}, orig: {xk0} ")
                return np.full_like(xi, np.nan)

    
    return np.full_like(xi, np.nan)


def update_graph(input, g):
    
    g.paths[:input.shape[0], :, 0] = input
    
    g._initialize_paths()


def func(input, env):
    input = np.array(input).reshape(-1, 2)
    
    update_graph(input, env)
    

    return -np.nan_to_num(env._get_reward()[0], nan=-1e10)


def cmaes(env, tolfun=0.001, sigma=0.00001):
    # TODO add bounds to system
    # es = cma.CMAEvolutionStrategy(g.nodes.flatten(), 0.1)
    # es.optimize(func) 

    # out = es.result.xbest.reshape(2,-1)
    # g_best = update_graph(out, deepcopy(g))

    # return g_best
    opt_env = deepcopy(env)
    opt_env.init_args['edges'] = env.get_edges()
    opts = {'bounds': [-1.0001, 1.0001], 'tolfun': tolfun}
    n = opt_env.number_of_nodes()
    es = cma.CMAEvolutionStrategy(opt_env.paths[:n,:,0].flatten(), sigma, inopts=opts)
    es.optimize(func, args=[opt_env]); 
    
    opt_env.paths[:n, :, 0] = es.result.xbest.reshape(-1,2)
    opt_env._initialize_paths()

    return opt_env