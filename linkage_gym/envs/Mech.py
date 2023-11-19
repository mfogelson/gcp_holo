import os
import sys
import pdb
from copy import deepcopy
from datetime import datetime
from itertools import combinations, product

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnchoredText

import gym
from gym import spaces

import networkx as nx
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from linkage_gym.utils.env_utils import distance, normalize_curve, parallel_method, rotate_points, symbolic_kinematics


class Mech(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}


    def __init__(self, max_nodes, bound, resolution, sample_points, feature_points, node_positions=None, edges=None, goal=None, normalize=True, self_loops=False, use_node_type=False, seed=None, fixed_initial_state=False, ordered_distance=False, constraints=[], min_nodes=5, debug=False, distance_metric='sqeuclidean'):
        """Initialize Mech gym environment

        Args:
            max_nodes (int): Maximum number of nodes (joints) in the linkage 
            bound (float): Bounds of the designs space [-bound, bound]
            resolution (int): Resolution of the scaffold nodes
            sample_points (int): Number of points that describe the node trajectories
            feature_points (int): Number of points that describe the node features for GNN
            node_positions (list/Nd.array, optional):  Shape: (n, 2) [x, y]. Defaults to None.
            edges (list/Nd.Array, optional): Shape: (e, 2) [id0, id1]. Defaults to None.
            goal (list/Nd.Array, optional): Shape: (2, T) Goal Trajectory. Defaults to None.
            normalize (bool, optional): Normalize the coupler trajectory with respect to the goal trajectory. Defaults to True.
            self_loops (bool, optional): Include self loops in adjaecency matrix for GNN. Defaults to False.
            use_node_type (bool, optional): Include binary indicator for GNN. Defaults to False.
            seed (int, optional): Seed for the gym environment. Defaults to None.
            fixed_initial_state (bool, optional): When reset is called, whether the original linkage stays the same or random valid linkage is sampled. Defaults to False.
            ordered_distance (bool, optional): Whether the points in the trajectory should be considered ordered. Defaults to False.
            constraints (list, optional): ([body constraints, coupler constraints]  [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]). Defaults to [].
            min_nodes (int, optional): Minimum number of nodes in the graph for a terminal action to be valid. Defaults to 5.
            debug (bool, optional): Enables some extra print statements. Defaults to False.
            distance_metric (str, optional): Check scipy.spatial.distance.cdist for various options. Defaults to 'sqeuclidean'.
        """

        super(Mech, self).__init__()

        self.debug = debug
        
        ## Random Seed
        self.rng_seed = None
        self.seed(seed)
        #np.random.seed(seed=seed)
        
        ## Keep same initial node and edges for training
        self.fixed_initial_state = fixed_initial_state 
        
        ## Design is terminal
        self.is_terminal = False
        
        ## Initialize Scaffold Nodes
        self.resolution = resolution
        self.bound = bound
        i = np.linspace(-bound, bound, resolution)
        ii, jj = np.meshgrid(i, i)
        self.grid = np.vstack([ii.flatten(), jj.flatten()]).T

        ## Environment Hyper parameters
        self.scaffold_ids = np.arange(self.resolution**2)
        self.T = sample_points
        self.feature_points = feature_points
        assert self.feature_points <= self.T 
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes 
        
        ## GCN hyper parameters
        self.normalize = normalize ## not used
        self.self_loops = self_loops
        self.use_node_type = use_node_type ## not used

        ## GYM Action Space and Observation Space
        self.node_combinations = tuple(combinations(range(max_nodes), 2))
        self.non_term_actions = tuple(product(self.node_combinations, range(self.resolution**2), [0])) # ncr(n, 2) | r^2 | {0, 1}
        self.term_actions = tuple(product(self.node_combinations, range(self.resolution**2), [1])) # ncr(n, 2) | r^2 | {0, 1}
        self.actions = self.non_term_actions+self.term_actions  #tuple(product(self.node_combinations, range(self.resolution**2), [0, 1])) # ncr(n, 2) | r^2 | {0, 1}
        self.num_actions = np.arange(len(self.actions))
        self.actions_hash = {k: v for k, v in zip(self.actions, self.num_actions)}
        self.non_term_actions_keys = [self.actions_hash[action] for action in self.non_term_actions]
        self.term_actions_keys = [self.actions_hash[action] for action in self.term_actions]

        self.valid_node_comb = {k: tuple(combinations(range(k), 2)) for k in range(self.max_nodes+1)}
        # self.actions = np.vstack([action_node.flatten(), action_pos.flatten()]).T
    
        self.action_space = spaces.Discrete(len(self.actions)) #self.node_combinations.shape[0]*self.resolution**2)
        self.max_length = 2*np.sqrt(2)*bound
        
        shape_x = self.max_nodes*(2*self.feature_points+int(use_node_type))
        shape_adj = self.max_nodes**2
        shape_mask = self.max_nodes
        shape_action_mask = len(self.actions)
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=[shape_x+shape_adj+shape_mask+shape_action_mask]) 

        ## Coupler Goal 
        self.goal = goal
        if goal.shape[0] != 2:
            goal = goal.T
        self.goal_scale = max(np.std(goal, axis=1).reshape(-1,1))
        self.goal_loc = np.mean(goal, axis=1).reshape(-1,1)
        
        self.distance_metric = distance_metric
        self.ordered = ordered_distance

        theta = np.linspace(0, np.pi*2, sample_points)
        circle = normalize_curve(np.array([np.cos(theta), np.sin(theta)]), scale=self.goal_scale, shift=self.goal_loc) #Shift and scale to goal size
        point = np.zeros([sample_points, 2])
        if goal is not None:
            self.R_circle = distance(goal, circle, self.ordered, self.distance_metric).sum()*(np.pi*2./self.T)
            self.R_point = distance(goal, point, self.ordered, self.distance_metric).sum()*(np.pi*2./self.T)
        # self.R_circle, _, _, _ = new_dist(goal, circle)
        # self.R_point, _, _, _ = new_dist(goal, point)
        self.goal_tol = 1e-6 ## Hyperparameter
        self.invalid_penalty = -100.0
        
        self.total_dist = None
        self.cycles = None
        self.reward = None
        
        ## Design Constraints 
        self.constraints = constraints # [body_const, coupler_const]
        
        ## Node Type 
        self.node_type = np.ones((self.max_nodes,1))
        self.node_type[0, 0] = 0 # Input node is fixed
        self.node_type[2, 0] = 0 # Fixed node

        ### Not fixed
        ## Mechanism Components
        self.paths = np.zeros([self.max_nodes, 2, self.T])
        self.adj = np.zeros([self.max_nodes, self.max_nodes])

        if node_positions is None:
            # edges = np.array([[0, 1], [0, 2]])
            # node_positions = np.array([[1.0, 0.7], [0.9, 0.7], [0.7, 0.8]])
            self.get_valid_env()
            node_positions = self.paths[:4, :, 0]
            edges = self.get_edges()
        else:
            n = node_positions.shape[0]
            self.paths[:n, :, 0] = node_positions
            self.adj[edges[:,0], edges[:,1]] = 1
            self.adj[edges[:,1], edges[:,0]] = 1

            ## Initialize Paths
            self._initialize_paths()

        self.previous_action = None

        self.best_designs = {}
        self.edge_masks = {}
        # action_mask = self._get_action_mask()
        self.prev_mask =  np.zeros(len(self.actions)) #action_mask
        
        self.init_args = {"node_positions": node_positions, 
                          "edges":          edges,} 
                        #   "prev_mask":      action_mask}
        self.resets = 0
        
        #! NEW 
        row_ind = np.arange(max(self.goal.shape))
        # row_inds = row_ind[row_ind[:,None]-np.zeros_like(row_ind)].T
        col_inds = row_ind[row_ind[:,None]-row_ind].T 
        self.index_variants = np.vstack([col_inds, col_inds[:,::-1]])
        
        
    def seed(self, seed=None):
        """Set the seed for gym env

        Args:
            seed (int, optional): seed used for gym env. Defaults to None.

        Returns:
            int: the seed value
        """
        if self.rng_seed:
            if self.debug:
                print("WARNING!!!! You tried reset the random number for the environment.")
                print("To overide the seed apply the following: \n", 
                      "env.rng_seed = seed \n" ,
                      "env.rng = RandomState(MT19937(SeedSequence(seed))) \n")
            return self.rng_seed
        self.rng_seed = seed
        self.rng = RandomState(MT19937(SeedSequence(seed))) 
        
        return seed
                        
    def clear_best_desings(self):
        """Clears self.best_designs
        """
        self.best_designs = {}
        return
        
    # @timebudget
    def _get_action_mask(self):
        """Finds all valid actions of the current environment state. NOTE: resuses previous masks if found in current episode

        Returns:
            Nd.Array: Binary vector of valid actions Shape: (num_actions, )
        """
        n = self.number_of_nodes()
        
        ## Reset all non-terminal actions to invalid if last action
        if n >= self.max_nodes-1:
            self.prev_mask[self.non_term_actions_keys] = 0
            
        
        node_combinations = self.valid_node_comb[n]

        for comb in node_combinations:
            ## Only if valid scaffold nodes not explored
            if comb not in self.edge_masks:
                ## Get valid scaffold node ids
                valid = self.get_edge_mask(comb[0], comb[1])
                
                ## Cache for future designs
                self.edge_masks[comb] = valid
            
            ## Update actions_mask
            for v in self.scaffold_ids[self.edge_masks[comb]]:
                ## If no constraints on coupler or body positions
                if self.constraints:
                     ## Check if scaffold node is in constraint TODO: this is wrong since it is trajectory and node position...
                    for i, c in enumerate(self.constraints):
                        if (i == 0 and n < self.max_nodes-1) or (i == 1 and n > self.min_nodes):
                            x_min, x_max, y_min, y_max = c
                            scaffold_x = self.grid[v,0]
                            scaffold_y = self.grid[v,1]
                            self.prev_mask[self.actions_hash[(comb, v, i)]] = float(scaffold_x > x_min and 
                                                                                    scaffold_x < x_max and
                                                                                    scaffold_y > y_min and 
                                                                                    scaffold_y < y_max)
                else:
                    ## Max number of actions
                    # if n < self.max_nodes-1:
                    self.prev_mask[self.actions_hash[(comb, v, 0)]] = 1
                    
                    ## Min number of actions
                    # if n > self.min_nodes:
                    self.prev_mask[self.actions_hash[(comb, v, 1)]] = 1
        ## 
        if n < self.min_nodes-1:
            self.prev_mask[self.term_actions_keys] = 0
            
        if n == self.max_nodes-1:
            self.prev_mask[self.non_term_actions_keys] = 0
            
        if self.previous_action:
            self.prev_mask[self.previous_action] = 0
    
        return self.prev_mask 
    
    def apply_random_action(self):
        """Applies a random valid action to the current environment

        Returns:
            (Nd.Array, float, bool, dict): Observation, Reward, Done, Info
        """
        action_mask = self._get_action_mask()

        if action_mask.sum() == 0.0:
            obs = self.get_observation()
            done = True 

            return obs, self.invalid_penalty, done, {} 
            
        action = self.rng.choice(self.num_actions, p=action_mask/action_mask.sum())
        
        return self.step(action)
    
    
    def reset(self):
        """Reset the environment to a root linkage

        Returns:
            Nd.Array: observation of the linkage (x, adj, mask, action_mask) flattened
        """

        ## Reset variables
        self.paths = np.zeros([self.max_nodes, 2, self.T])
        self.adj = np.zeros([self.max_nodes, self.max_nodes])
        self.edge_masks = {}
        self.prev_mask = np.zeros(len(self.actions)) 
        self.previous_action = None
        self.is_terminal = False
        
        self.resets += 1
        
        ## Get root node design
        if self.fixed_initial_state:
            node_positions = self.init_args['node_positions']
            edges = self.init_args['edges']

            n = node_positions.shape[0]

            self.paths[:n, :, 0] = node_positions

            self.adj[edges[:,0], edges[:,1]] = 1
            self.adj[edges[:,1], edges[:,0]] = 1

            self._initialize_paths()
            
        else:
            self.get_valid_env()
        
        return self.get_observation()

    def _get_fixed_ids(self):
        """Returns the indexes of revolute joints that are fixed

        Returns:
            Nd.Array: indexes of fixed nodes Shape:(m, )
        """

        return np.argwhere(self.node_type == 0)[:,0]

    def _get_crank_id(self):
        """Returns the index for the linkage connected to the motor input

        Returns:
            int: index of crank node
        """

        return 1 # NOTE: np.argwhere(self.adj[0,:] == 1).item()

    def _get_dist(self, p1, p2):
        """Helper function to get the distance between two points

        Args:
            p1 (Nd.Array): Point1 Shape: (2,)
            p2 (Nd.Array): Point2 Shape: (2,)

        Returns:
            float: Distance between p1 and p2
        """

        return np.linalg.norm(p1-p2)

    def _get_angle(self, p1, p2):
        """Helper function to get the angle between two points

        Args:
            p1 (Nd.Array): Point1 Shape: (2,)
            p2 (Nd.Array): Point2 Shape: (2,)

        Returns:
            float: Angle between vectors from origin to p1 and p2
        """

        return np.arctan2(*(p2[::-1]-p1[::-1])) % (2*np.pi)

    def _update_crank_path(self):
        """Helper function to update self.paths the trajectory of the crank revolute joint
        """

        crank_id = self._get_crank_id()
        edge_length = self._get_dist(self.paths[0,:,0], self.paths[crank_id, :, 0])
        start_pos = self.paths[crank_id, :, 0]
        start = self._get_angle(self.paths[0,:,0], self.paths[crank_id, :, 0]) 
        try:
            assert (np.cos(start)*edge_length+self.paths[0,0,0])-start_pos[0] <= 1e-3 and (np.sin(start)*edge_length+self.paths[0,1,0])-start_pos[1] <= 1e-3
        except:
            pdb.set_trace()
        theta = np.linspace(start, start+(np.pi*2), num=self.T)
        
        # pdb.set_trace()
        self.paths[crank_id, :, :] = np.array([np.cos(theta), np.sin(theta)])*edge_length + self.paths[0,:,0].reshape(-1,1)

    def _initialize_paths(self):
        """Updates self.paths with trajectories of current linkage
        """

        # Initialize fixed node positions
        fixed_ids = self._get_fixed_ids()
        self.paths[fixed_ids, :, :] = self.paths[fixed_ids, :, 0][:, :, np.newaxis]

        # Initialize pin node positions
        self._update_crank_path()
        
        self.update_paths() 
    
    def number_of_nodes(self):
        """Helper function returns number of nodes currently in the linkage graph

        Returns:
            int: number of nodes
        """
        empty_rows = np.argwhere(self.adj.sum(1) == 0)
        if len(empty_rows) > 0:
            return empty_rows[0].item()
        else:
            return self.max_nodes

    def number_of_edges(self):
        """Helper function returns the number of edges that make up the current linkage graph

        Returns:
            int: number of edges
        """
        
        return sum(sum(self.adj))//2

    def get_edges(self, limit=None):
        """Helper function to return all the edges in the current linkage graph

        Args:
            limit (int, optional): edges between nodes bellow a particular index. Defaults to None.

        Returns:
            Nd.Array: Array of edge index pairs Shape: (e, 2) [id0, id1]
        """
        
        if limit is None:
            limit = self.max_nodes

        return np.array([[i, j] for i in range(self.max_nodes) for j in range(i) if self.adj[i,j]])

    def get_edge_lengths(self):
        """Helper function to return all the edge lengths

        Returns:
            Nd.Array: Lengths of each edge Shape: (e, )
        """
        
        edges = self.get_edges()
        lengths = np.zeros([edges.shape[0],])
        
        for i, e in enumerate(edges):
            lengths[i] = self._get_dist(self.paths[e[0],:,0], self.paths[e[1], :, 0])

        return lengths

    def update_paths(self, unknown_joints=None):
        """Update self.paths

        Args:
            unknown_joints (list, optional): node indexes that are not known or want to be calculated. Defaults to None.
        """
        n = self.number_of_nodes()
        
        if unknown_joints is None:
            known_joints = list(np.argwhere(self.node_type == 0)[:,0])
            known_joints.append(1) #(np.argwhere(self.adj[0,:] == 1).item()) #TODO: Fix this
            unknown_joints = list(set(range(n)) ^ set(known_joints)) 
        else:
            assert isinstance(unknown_joints, list)
            known_joints = list(set(range(n)) ^ set(unknown_joints))


        count = 0
        while list(set(range(n)) ^ set(known_joints)) != [] and count < 100:

            for i in unknown_joints[:]:
                
                if sum(self.adj[i, known_joints]) >= 2:


                    inds = np.array(known_joints)[np.where(self.adj[i, known_joints] >= 1)[0]]

                    # Update paths
                    self.paths[i, :, :] = symbolic_kinematics(self.paths[inds[0],:,:], self.paths[inds[1], :, :], self.paths[i, :, 0])
                    
                    unknown_joints.remove(i)
                    known_joints.append(i)
                else:
                    pass
            count += 1
        
        
    def add_node(self, node_pos):
        """Add node to the linkage graph

        Args:
            node_pos (Nd.Array): initial position of revolute joint Shape: (2, )
        """
        
        n = self.number_of_nodes()
        
        self.paths[n, :, 0] = node_pos

    def add_edge(self, id0, id1):
        """Add edge to linkage graph

        Args:
            id0 (int): index of node 0
            id1 (int): index of node 1
        """
        
        self.adj[id0, id1] = 1
        self.adj[id1, id0] = 1

    def update_fixed_paths(self, fixed_node_pos):
        """Update self.paths for fixed revolute joints (DEPRECATED)

        Args:
            fixed_node_pos (Nd.Array): Vector of initial position of fixed nodes Shape: (n, 2)
        """
        fixed_ids = self._get_fixed_ids()
        self.paths[fixed_ids, :, :] = fixed_node_pos[:, :, np.newaxis]

        self._update_crank_path()

        # Update rest of mechanism
        self.update_paths()

    def coupler_traj(self, normalize=True, scale=None, shift=None):
        """Return the coupler node trajectory

        Args:
            normalize (bool, optional): Normalized curve. Defaults to True.
            scale (float, optional): scaling factor. Defaults to None.
            shift (Nd.Array, optional): x,y shift for all points. Defaults to None.

        Returns:
            Nd.Array: coupler trajectory
        """
        n = self.number_of_nodes()
        # inds = np.linspace(0, self.T-1, self.test_samples).astype(int)

        if normalize: return normalize_curve(self.paths[n-1, :, :], scale=scale, shift=shift)
        
        return self.paths[n-1, :, :]


    def paper_plotting(self, show=False, show_goal=True, show_coupler=True, show_obj=True):
        """Helper function for plotting figures used in paper

        Args:
            show (bool, optional): show the plot. Defaults to False.
            show_goal (bool, optional): plot the goal on figure. Defaults to True.
            show_coupler (bool, optional): plot the coupler curve on figure. Defaults to True.
            show_obj (bool, optional): add objective value of linkage to figure. Defaults to True.

        Returns:
            Matplotlib.fig: the figure object
        """
        
        fig, ax1 = plt.subplots(figsize=(8.5, 11))
        coupler_idx = self.number_of_nodes()-1
        
        ax1.plot(self.grid[:,0], self.grid[:,1], 'k.', ms=4)

        if self.previous_action:
            ax1.plot(self.grid[self.edge_masks[self.actions[self.previous_action][0]],0], self.grid[self.edge_masks[self.actions[self.previous_action][0]],1], 'go', ms=4)
            ax1.plot(self.grid[~self.edge_masks[self.actions[self.previous_action][0]],0], self.grid[~self.edge_masks[self.actions[self.previous_action][0]],1], 'rx', ms=4)

        ## Plot Links and Joints
        edges = self.get_edges()
        for e in edges: 
            ax1.plot(self.paths[e, 0, 0],self.paths[e,1, 0], '-', color='0.7', linewidth=3, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]) 
            # plt.plot(self.paths[e, 0, 0],self.paths[e,1, 0], 'r.', label="joints")
        
        ## Plot special joints    
        ax1.plot(self.paths[2, 0, 0], self.paths[2,1, 0], marker='^', color='gray', label="fixed joint", ms=15, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
        ax1.plot(self.paths[0, 0, 0], self.paths[0,1, 0], marker='^', color='magenta', label="motor joint", ms=15, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
        ax1.plot(self.paths[1, 0, 0], self.paths[1,1, 0], marker='o', color='lime', label="crank joint", ms=15)

        ## Plot moveable revolute joints
        fixed_ids = self._get_fixed_ids()
        non_fixed_ids = list(set(fixed_ids) ^ set(range(coupler_idx+1)))
        # for n in non_fixed_ids[1:]:
        #     ax1.plot(self.paths[n, 0, :], self.paths[n,1, :], 'b-', label="pin path", ms=4)
        ax1.plot(self.paths[non_fixed_ids[1:], 0, 0], self.paths[non_fixed_ids[1:],1, 0], 'ro', label="pin joints", ms=15)

        ## Plot Coupler Path 
        if show_coupler:
            ax1.plot(self.paths[coupler_idx, 0, :], self.paths[coupler_idx, 1,:], 'b-', label="coupler", linewidth=4)
        ax1.plot(self.paths[coupler_idx, 0, 0],self.paths[coupler_idx, 1, 0], 'yo', label="coupler joint", markersize=15)

        ## Plot Shifted Goal
        mu = self.paths[coupler_idx, :, :].mean(1).reshape(-1, 1)
        std = max(self.paths[coupler_idx, :, :].std(1))
        traj_norm = self.coupler_traj(scale=self.goal_scale, shift=self.goal_loc) 
        total_dist, theta, _ = parallel_method(self.index_variants, self.goal, traj_norm)
        
        goal = (rotate_points(normalize_curve(self.goal).T, theta).T*std+mu)
        # goal = normalize_curve(self.goal)*std+mu
        if show_goal:
            ax1.plot(goal[0,:], goal[1,:], 'y-', linewidth=4)
        
        ## Plot constraints    
        if self.constraints:
            # plt.axhline(y=0, color='red', linestyle='--', lw=5)

            body_constraints, coupler_constraints = self.constraints
            ax1.plot([body_constraints[0], body_constraints[1], body_constraints[1], body_constraints[0], body_constraints[0]], 
                     [body_constraints[2], body_constraints[2], body_constraints[3], body_constraints[3], body_constraints[2]], 'r-.', lw=4)
            
            ax1.plot([coupler_constraints[0], coupler_constraints[1], coupler_constraints[1], coupler_constraints[0], coupler_constraints[0]], 
                     [coupler_constraints[2], coupler_constraints[2], coupler_constraints[3], coupler_constraints[3], coupler_constraints[2]], 'g-.', lw=4)
        ax1.set_axis_off()
        
        ## Add Objective score to figure
        if show_obj:
            # traj_norm = self.coupler_traj(scale=self.goal_scale, shift=self.goal_loc) 

            # total_dist = distance(goal, self.paths[coupler_idx, :, :], ordered=self.ordered, distance_metric=self.distance_metric).sum()*(np.pi*2./self.T)
            
            #! total_dist = distance(self.goal, traj_norm, ordered=self.ordered, distance_metric=self.distance_metric).sum()*(np.pi*2./self.T)
            #! New
            total_dist = self.get_distance(scale=self.goal_scale, shift=self.goal_loc)
            # pdb.set_trace()
            plt.rcParams['font.size'] = 40
            text_box = AnchoredText(f"obj={round(total_dist, 2)}", frameon=False, pad=0.0, borderpad=-1.0, loc='lower right')
            # text_box = AnchoredText(f"obj_MICP={round(total_dist2, 2)}, obj_GCPN={round(total_dist, 2)}", frameon=False, pad=0.0, borderpad=-1.0, loc='lower right')
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)

            ax1.add_artist(text_box)


        ax1.set_xlim([-1.7, 2.5])
        ax1.set_ylim([-1.7, 1.7])
        ax1.set_aspect('equal')

    
        if show:
            plt.show()
            
        return fig
    
    def coords_to_pix(self, x):
        """Helper function for self.render converting coordinates to pixels

        Args:
            x (Nd.Array): coordinates Shape: (2, n)

        Returns:
            Nd.Array: pixel locations Shape: (2, n)
        """
        return x * self.scale + self.screen_width / 2.0
    
    def render(self, mode="human"):
        """Render the linkage being generated

        Args:
            mode (str, optional): visualization mode. Defaults to "human".

        Raises:
            DependencyNotInstalled: needs pygame

        Returns:
            bool: successful
        """
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        self.screen_width = 600
        self.screen_height = 600

        design_width = self.bound * 5.0
        self.scale = self.screen_width/ design_width 


        self.screen = None # TODO: Fix this
        self.clock = None # TODO: Fix me
        

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        
        path_pix = self.coords_to_pix(self.paths).astype(int)
        
        
        ## Draw Links
        for e in self.get_edges():
            gfxdraw.line(self.surf, path_pix[e[0], 0, 0], path_pix[e[0], 1, 0], 
                                    path_pix[e[1], 0, 0], path_pix[e[1], 1, 0], [0, 0, 0])
        
        rad = 5
        ## Draw Motor
        gfxdraw.filled_circle(self.surf, path_pix[0,0,0], path_pix[0,1,0], rad, [255, 0, 255])
        
        ## Draw Crank Node
        gfxdraw.filled_circle(self.surf, path_pix[1,0,0], path_pix[1,1,0], rad, [0, 255, 0])
        
        ## Draw Fixed Node
        gfxdraw.filled_circle(self.surf, path_pix[2,0,0], path_pix[2,1,0], rad, [160, 160, 160])
        
        ## Draw Other Nodes
        n = self.number_of_nodes()
        for i in range(3, n):
            gfxdraw.filled_circle(self.surf, path_pix[i,0,0], path_pix[i,1,0], rad, [255, 0, 0])

        
        ## Draw Coupler Traj
        for i in range(self.T):
            gfxdraw.filled_circle(self.surf, path_pix[n-1,0,i], path_pix[n-1,1,i], rad//2, [255, 0, 0])

        mu = self.paths[n-1, :, :].mean(1).reshape(-1, 1)
        std = max(self.paths[n-1, :, :].std(1))
        # pdb.set_trace()
        goal = (self.goal*std+mu)
        goal_pix = self.coords_to_pix(goal).astype(int)
        ## Draw Goal Traj
        for i in range(self.T):
            gfxdraw.filled_circle(self.surf, goal_pix[0,i], goal_pix[1,i], rad//2, [255, 255, 0])
        

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return True


    def plot_graph(self, plot_paths=False, plot_coupler=True, filename=None, coupler_idx=None):
        """Helper function to visualize linkage graph

        Args:
            plot_paths (bool, optional): plot revolute joint trajectories. Defaults to False.
            plot_coupler (bool, optional): plot coupler joint trajectory. Defaults to True.
            filename (str, optional): filename to save figure. Defaults to None.
            coupler_idx (int, optional): index of coupler index or other node that you want to be known as the coupler. Defaults to None.

        Returns:
            Matplotlib.fig: figure 
        """
        ## Get coupler index
        if coupler_idx is None: 
            coupler_idx = self.number_of_nodes()-1
        
        ## Initialize figure    
        if self.goal is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            fig, ax1 = plt.subplots()
        
        ## Plot Edges
        edges = self.get_edges(coupler_idx+1)
        for e in edges: 
            ax1.plot(self.paths[e, 0, 0],self.paths[e,1, 0], 'k-') 
            # plt.plot(self.paths[e, 0, 0],self.paths[e,1, 0], 'r.', label="joints")
        
        ## Plot motor and fixed node
        ax1.plot(self.paths[2, 0, 0], self.paths[2,1, 0], 'r^', label="fixed joints", ms=10)
        ax1.plot(self.paths[0, 0, 0], self.paths[0,1, 0], 'm^', label="motor joints", ms=10)

        ## Plot revolute joints
        fixed_ids = self._get_fixed_ids()
        non_fixed_ids = list(set(fixed_ids) ^ set(range(coupler_idx+1)))
        ax1.plot(self.paths[non_fixed_ids, 0, 0], self.paths[non_fixed_ids,1, 0], 'ro', label="pin joints", ms=10)

        ## Plot joint trajectories
        if plot_paths:
            ax1.plot(self.paths[:coupler_idx+1, 0, :], self.paths[:coupler_idx+1, 1,:], 'b.', markersize=1.0)

        ## Highlight coupler trajectory
        if plot_coupler:
            # n = self.number_of_nodes()-1
            ax1.plot(self.paths[coupler_idx, 0, 0],self.paths[coupler_idx, 1, 0], 'yo', label="coupler joint", markersize=10)
            ax1.plot(self.paths[coupler_idx, 0, :], self.paths[coupler_idx, 1,:], 'y-', label="coupler", markersize=3)


        ## Figure Formating
        ax1.legend(loc='upper center', bbox_to_anchor=(1.0, -0.1),
            fancybox=True, shadow=True, ncol=3)
        # ax1 = plt.gca()
        ax1.set_title('State Visualization')
        ax1.set_xlim([-self.max_length, self.max_length])
        ax1.set_ylim([-self.max_length, self.max_length])
        ax1.set_aspect('equal')

        ## Plot goal
        if self.goal is not None:
            ax2.plot(self.goal[0,:], self.goal[1,:], 'r.', ms=10)
            coupler = self.coupler_traj()
            ax2.plot(coupler[0,:], coupler[1,:], 'y.')
            ax2.set_xlim([-self.max_length, self.max_length])
            ax2.set_ylim([-self.max_length, self.max_length])
            ax2.set_aspect('equal')

        
        ## Plot constraints
        if self.constraints:
            colors = ['r--', 'b--']
            for i, c in enumerate(self.constraints):
                ax1.plot([c[0], c[1], c[1], c[0]], [c[2], c[2], c[3], c[3]], colors[i], lw=3)

        ## Save figure
        if filename is not None: 
            if filename in os.listdir(): 
                out = filename.split(".")
                out[0] += ('_'+str(datetime.now()))
                filename = '.'.join(out)
            
            plt.savefig(filename)
            plt.close()
        
        return fig

    def is_valid(self):
        """Checks that linkage graph is valid

        Returns:
            bool: linkage graph is valid
        """
        return (not np.isnan(self.paths).any())
            
    
    # @timebudget
    def get_edge_mask(self, id0, id1):
        """Valid scaffold node locations for adding to Assur 0DOF linkage to node0 and node1

        Args:
            id0 (int): index of node 0
            id1 (int): index of node 1

        Returns:
            Nd.Array: valid scaffold nodes for adding to linkage graph
        """
        ## Get nodei and nodej trajectories
        xi = self.paths[id0, :, :] # 2xsteps
        xj = self.paths[id1, :, :]

        l_ij = np.linalg.norm(xi - xj, axis=0).reshape(1, -1) # (1, steps)

        l_ik = np.linalg.norm(xi[:, 0].reshape(1,-1) - self.grid, axis=1).reshape(-1, 1) # [1,2] - [121, 2] -> (121, 1)
        l_jk = np.linalg.norm(xj[:, 0].reshape(1,-1) - self.grid, axis=1).reshape(-1, 1) # [1,2] - [121, 2] -> (121, 1)

        ## Triangle inequality between node i and j trajectories
        valid = np.logical_and.reduce((np.all(l_ik+l_jk > l_ij, 1), 
                                        np.all(l_ik+l_ij > l_jk, 1),  
                                        np.all(l_jk+l_ij > l_ik, 1)))

        return valid #np.ones_like(valid)

    
    def satisfy_constraints(self): 
        """Checks if linkage graph satisfies the constraints

        Returns:
            bool: satifies
        """
            
        if not self.constraints:
            return True
        
        valid = []
        for i, bounding_box in enumerate(self.constraints):
            ## Get body and couple index
            if self.is_terminal:
                node = self.number_of_nodes()-1 if i else range(self.number_of_nodes()-1)
            else:
                node = range(self.number_of_nodes())
                
                ## If non-terminal and coupler constraints break
                if i:
                    return all(valid)
            
            ## Check that iniital state inside bounding box 
            # TODO: this is wrong, it should consider the whole trajectory
            if not bounding_box:
                valid.append(True)
            
            xmin = np.min(self.paths[node,0,:])
            xmax = np.max(self.paths[node,0,:])
            
            ymin = np.min(self.paths[node,1,:])
            ymax = np.max(self.paths[node,1,:])
            
            valid.append(all([xmin>bounding_box[0], 
                                xmax<bounding_box[1], 
                                ymin>bounding_box[2], 
                                ymax<bounding_box[3]]))
                
        return all(valid)
        
    
    def get_distance(self, scale=None, shift=None):
        """Distance between the coupler trajectory and the goal

        Args:
            scale (float, optional): scaling factor for coupler trajectory. Defaults to None.
            shift (Nd.Array, optional): shifting vector for coupler trajectory. Defaults to None.

        Returns:
            float: distance
        """
        ## Dist
        traj_norm = self.coupler_traj(scale=scale, shift=shift) 
        #! NEW
        total_dist, _, _ = parallel_method(self.index_variants, self.goal, traj_norm)
        total_dist *= (np.pi*2./self.T)
        # total_dist = distance(self.goal, traj_norm, ordered=self.ordered, distance_metric=self.distance_metric).sum()*(np.pi*2./self.T)
        
        return total_dist

    def _get_reward(self):
        """Reward for linkage graph

        Returns:
            (float, bool): reward, success
        """
        
        self.cycles = self.number_of_cycles()

        ## Not Valid
        if not self.is_valid():
            self.total_dist = np.nan
            self.reward = self.invalid_penalty
            return self.reward, False
        
        ## Fails Constraints
        if not self.satisfy_constraints(): 
            self.total_dist = np.nan
            self.reward = self.invalid_penalty
            return self.reward, False

        ## No goal
        if self.goal is None:             
            print("[Warning] _get_reward(): No goal was added")
            return 0.0, False
        
        ## Invalid design
        if self.cycles == 0:
            self.total_dist = np.nan
            self.reward = self.invalid_penalty
            return self.reward, False
        
        ## get distance
        self.total_dist = self.get_distance(scale=self.goal_scale, shift=self.goal_loc)
        
        ## Normalize distance w.r.t circle 
        norm_distance_reward = -self.total_dist #max((self.R_circle-self.total_dist)/self.R_circle, -0.0) # 0-1
        
        ## set reward
        self.reward = norm_distance_reward #max(-self.total_dist, -9.9) #(cycle_weight*cycle_reward + (1.-cycle_weight)*norm_distance_reward)*10.0
        
        return self.reward, (self.total_dist <= self.goal_tol) 
        

    def _get_info(self):
        """linkage graph information

        Returns:
            dict: various information that might be useful
        """
        
        ## Only return info if terminal linkage design
        if self.is_terminal:
            n = self.number_of_nodes()
            n_active = self.number_of_active_nodes()
            
            info = {'number_of_nodes': n,
                    'number_of_active_nodes': n_active, 
                    'max_nodes':       self.max_nodes,
                    'resolution':      self.resolution,
                    'bound':           self.bound,
                    'feature_points':  self.feature_points,
                    'sample_points':   self.T,
                    'number_of_edges': self.number_of_edges(),
                    'node_positions':  self.paths[:n, :, 0],
                    'edges':           self.get_edges(),
                    'valid':           self.is_valid(),
                    'goal':            self.goal,
                    'coupler':         self.paths[n-1, :, :],
                    'reward':          self.reward, ## This includes information that is biased about desired behavior
                    'cycles':          self.cycles,
                    'distance':        self.total_dist, ## This is the actual metric of comparison
                    }
            
            return info #info
        return {}
    

    def _remove_action(self):
        """Helper function that removes previous action
        """
        n = self.number_of_nodes()
        self.paths[n-1, :, :] = 0.

        self.adj[n-1,:] = 0
        self.adj[:,n-1] = 0
        
        m = self.number_of_nodes()
        assert((n-m)==1)
        assert(not np.isnan(self.paths).any())


    def dfs(self, visited, edges, node, known_joints):
        """Helper function for depth first search

        Args:
            visited (list): visited nodes
            edges (set): set of edges
            node (int): node index
            known_joints (list): known joint trajectories
        """
        
        if node not in visited and node not in [0, 1, 2]:
            visited.add(node)

            neighbours = np.array(known_joints)[np.where(self.adj[int(node), known_joints] >= 1)[0]][:2]
            # print(node, neighbours)
            edges.add(frozenset((node, neighbours[0])))
            edges.add(frozenset((node, neighbours[1])))
            for neighbour in neighbours:
                self.dfs(visited, edges, neighbour, known_joints)
                
    def get_active_nodes(self):
        """Helper function to get all known nodes that contribute to coupler trajectory

        Returns:
            (list, set): node indexes that are used for coupler FK, edges that are useful for linkage graph
        """
        ## All nodes
        n = self.number_of_nodes()

        ## Initialize variables
        visited = set()
        edges = set()
        known_joints = np.arange(n)

        ## Recursively trace from coupler to root nodes 
        self.dfs(visited, edges, n-1, known_joints)
        visited = list(visited)
        visited.sort()

        active_nodes = [0, 1, 2] + visited
        
        return active_nodes, edges
    
    def number_of_active_nodes(self):
        """Helper function returns number of active nodes

        Returns:
            int: number of active nodes in linkage graph
        """
        return len(self.get_active_nodes()[0])
    
    def prune(self):
        """Prune linkage graph of unnecessary revolute joints
        """
        
        ## Get active nodes and edges
        active_nodes, edges = self.get_active_nodes()
        active_nodes.sort()
        
        n = len(active_nodes)
        
        ## update edge list
        edges = np.array([[0, 1], [0, 2]]+ [[active_nodes.index(list(e)[0]), active_nodes.index(list(e)[1])] for e in edges])
        
        ## get all paths of active nodes
        paths = self.paths[active_nodes,:,0]

        ## Reset linkage graph paths and adj
        self.paths = np.zeros_like(self.paths)
        self.adj = np.zeros_like(self.adj)
        
        ## Update with only active nodes and edges
        self.paths[:n, :, 0] = paths
        
        self.adj[edges[:,0], edges[:,1]] = 1
        self.adj[edges[:,1], edges[:,0]] = 1 
     
        ## reinitialize linkage graph
        self._initialize_paths()
        
    def active_cycles(self):
        """Returns all the active loops in the linkage graph  

        Returns:
            list: all cycles in linkage graph NOTE: this includes cycles with 3 nodes which are not valid loops
        """
        
        ## Get active nodes and edges
        active_nodes, edges = self.get_active_nodes()
        active_nodes.sort()
        
        n = len(active_nodes)
        
        edges = np.array([[0, 1], [0, 2]]+ [[active_nodes.index(list(e)[0]), active_nodes.index(list(e)[1])] for e in edges])
        
        ## initialize nx graph
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)
        
        ## Minimum cycle basis
        return list(nx.minimum_cycle_basis(graph))
    
    def number_of_cycles(self):
        """Number of active linkage graph loops

        Returns:
            int: number of active linkage graph loops NOTE: this excludes loops of 3 nodes (also known as triads as they are not useful)
        """
        cycles = [c for c in self.active_cycles() if len(c) > 3]
        
        return len(cycles)
        

    def update_best_designs(self):
        """Update set of best designs of various linakge graph topologies
        """
        ## Prune linkage graph
        self.prune()
        
        ## If linkage graph topology not accounted for yet        
        if self.cycles > 0:
            if self.cycles not in self.best_designs:
                self.best_designs[self.cycles] = (deepcopy(self.paths[:,:,0]), deepcopy(self.get_edges()), deepcopy(self.reward))
                return 
            
            ## If linkage graph is better than current topology 
            if self.reward > self.best_designs[self.cycles][-1]:
                self.best_designs[self.cycles] = (deepcopy(self.paths[:,:,0]), deepcopy(self.get_edges()), deepcopy(self.reward))
                return 

    def step(self, action):
        """Update linkage graph with new action

        Args:
            action (int): index of action 

        Returns:
            (Nd.Array, float, bool, dict): Observation, Reward, Done, Info
        """
        if self.debug:
            if self.prev_mask[int(action)] == 0:
                print(sum(self.prev_mask))
                print("selected an action that was deemed invalid")
        ## Get action from index
        (node_id0, node_id1), scaffold_id, done = self.actions[int(action)]
        
        ## Is terminal action
        self.is_terminal = done


        ## If Action is the same as previous Terminate Episode
        if action == self.previous_action: 
            if self.debug:
                print("Warning: (Action) Same Action was selected again. Note that this is considered invalid")
                
            obs = self.get_observation()
            done = True 

            return obs, self.invalid_penalty, done, {} 

        
        ## Check if node selection is valid 
        n = self.number_of_nodes()
        if node_id0 >= n or node_id1 >= n:
            if self.debug:
                print("Warning: (Nodes) Same Action was selected again. Note that this is considered invalid")

            return self.get_observation(), self.invalid_penalty, True, {}

        ## Valid node add to linkage
        self.add_node(self.grid[scaffold_id, :])
        new_node_id = self.number_of_nodes()
        self.add_edge(node_id0, new_node_id)
        self.add_edge(node_id1, new_node_id)

        ## Update Paths
        self.paths[new_node_id, :, :] = symbolic_kinematics(self.paths[node_id0, :, :], self.paths[node_id1, :, :], self.paths[new_node_id, :, 0])

        ## If Kinematics Valid Check
        if not self.is_valid():
            if self.debug:
                print("Warning: (Kinematics) Action led to kinematically infeasible design.")

            self._remove_action()
            return self.get_observation(), self.invalid_penalty, True, {}
        
        ## Check if finished design 
        if self.number_of_nodes() == self.max_nodes and not done:
            if self.debug:
                print("Warning: (Terminal) Failed to finish design in valid number of steps.")

            return self.get_observation(), self.invalid_penalty, True, {}

        ## Get reward
        reward = 0.0
        if done:
            reward, solved = self._get_reward()
            if solved:
                print("Found exact solution...stop search")
            ## Save all good designs during the search
            self.update_best_designs()
    
        self.previous_action = action
        
        obs = self.get_observation()
        info = self._get_info() 
        
        # Return Status
        return obs, reward, done, info 
    
    def get_observation(self):
        """Observation of current linkage state

        Returns:
            Nd.Array: [X ((Node_features)*max_nodes), adj (max_nodes*max_nodes), mask (max_nodes), action_mask (number_of_actions)] flattened
        """
        obs = []

        ## Revolution joint positons 
        idx = np.round(np.linspace(0, self.paths.shape[-1]-1, self.feature_points)).astype(int)
        
        ## NOTE: use_node_type DEPRECATED
        # if self.use_node_type:
        #     x = [[self.paths[i, 0, 0], self.paths[i, 1, 0], 1] if (i == self.number_of_nodes()-1 and self.is_terminal) else 
        #         [self.paths[i, 0, 0], self.paths[i, 1, 0], 0] for i in range(self.max_nodes)] 
        #     obs.append(np.asarray(sum(x, [])).astype('float32'))

        # else:
        #     # x = [[self.paths[i, 0, 0], self.paths[i, 1, 0]] for i in range(self.max_nodes)]
        
        ## Node features
        x = self.paths[:, :, idx].flatten().astype('float32')
        obs.append(x)
        
        ## Adjacency Matrix
        adj = self.adj.copy()
        if self.self_loops:
            np.fill_diagonal(adj, 1.0)
            
        obs.append(adj.astype('float32').flatten())

        ## Node Mask 
        n = self.number_of_nodes()
        mask = np.zeros(self.max_nodes).astype(int)
        mask[:n] = 1
        obs.append(mask.astype('float32'))

        ## Node Action Mask
        action_mask = np.array(self._get_action_mask())
        obs.append(action_mask.astype('float32'))
        
        obs = np.concatenate(obs)

        return np.nan_to_num(obs, nan=0.0)
    
    def random_4_bar(self, edges, bar_type=None):
        """Random valid n bar linakge NOTE: not really N_bar, based on edges input

        Args:
            edges (Nd.Array): set of edges Shape: (e, 2)
            n (int): number of revolute joints
        """
        pos_ind = range(self.grid.shape[0])
        # s = l = 1 
        # p = q = 0
        
        # ## Random Crank-Rocker N-Bar
        # while s+l > p+q:
        node_pos_ind = self.rng.choice(pos_ind, size=4, replace=False)
        node_pos = self.grid[node_pos_ind,:]
        
        lengths = [np.linalg.norm(node_pos[0,:]- node_pos[1,:]), np.linalg.norm(node_pos[1,:]- node_pos[3,:]), np.linalg.norm(node_pos[3,:]- node_pos[2,:]), np.linalg.norm(node_pos[2,:]- node_pos[0,:])]
        if all(lengths[0] < lengths[1:]):
            if bar_type == 1:
                return False
            ## Crank Rocker
            if any(lengths[1] < [lengths[2], lengths[3]]):
                return False

            if (lengths[0]+lengths[1] > lengths[2]+lengths[3]):
                # print("not valid")
                return False
            
            self.base_type = 0
        elif all(lengths[0] > lengths[1:]):
            if bar_type == 0:
                return False
            ## Double Rocker
            if any(lengths[3] > [lengths[1], lengths[2]]):
                return False
            if (lengths[0]+lengths[3] > lengths[1]+lengths[2]):
                # print("not valid")
                return False
            # print("valid double rocker")
            self.base_type = 1
        else:
            return False
            # s = np.linalg.norm(node_pos[0]-node_pos[1])
            # q = np.linalg.norm(node_pos[0]-node_pos[2])
            # l = np.linalg.norm(node_pos[1]-node_pos[3])
            # p = np.linalg.norm(node_pos[2]-node_pos[3])
        
        
        self.paths[:4, :, 0]=node_pos
        self.adj[edges[:,0], edges[:,1]] = 1
        self.adj[edges[:,1], edges[:,0]] = 1 
        self._initialize_paths()
        return True

    def get_valid_env(self, bar_type=None):
        """Generate random valid linkage graph

        Returns:
            bool: is valid
        """

        ## Basic 4-bar configuration
        edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        self.random_4_bar(edges, bar_type)
        
        valid = False
        # Step 2 check validity 
        while not self.is_valid() or not self.satisfy_constraints() or not valid:
            # If not valid save data
            valid = self.random_4_bar(edges, bar_type)


        return self.is_valid()
        
        
    def close(self):
        """Close environment
        """
        pass




if __name__ == "__main__":
    import pyvirtualdisplay

    env = Mech(max_nodes=10, bound=1., resolution=11, sample_points=20, feature_points=1, goal=np.zeros([2,20]), normalize=True)
    _display = pyvirtualdisplay.Display(visible=True,  # use False with Xvfb
                                    size=(600, 600))
    _ = _display.start()
    done = False
    while not done:
        env.render()
        _, _, done, _ = env.apply_random_action()
    _display.stop()
    # # pdb.set_trace()
    # filenames = ['jansen_traj', 'klann_traj', 'strider_traj', 'trot_traj'] #['jansen_traj', 'klann_traj'] #, 'strider_traj', 'trot_traj'] #, 'infinity_200', 'loopfolium_150', 'quadrifoilium_200', 'trifolium_100', 'double_loopfolium_200']
    # sample_points = 20
    
    # for goal_filename in filenames:
    #     for _ in range(1):
    #         # goal_filename = "fake_jansen_goal"
    #         tb_log_dir = f"./logs/{goal_filename}"

    #         if not os.path.isdir(tb_log_dir):
    #             os.mkdir(tb_log_dir)

    #         goal_curve = pickle.load(open(f'saved_footpaths/{goal_filename}.pkl', 'rb'))
    #         idx = np.round(np.linspace(0, goal_curve.shape[1] - 1, sample_points)).astype(int)
    #         goal = normalize_curve(goal_curve[:,idx])

    #         env_kwargs = {"max_nodes":10, 
    #                       "bound":1., 
    #                       "resolution":11, 
    #                       "sample_points":sample_points, 
    #                       "goal":goal, 
    #                       "normalize":True, 
    #                       "seed": 2, 
    #                       "fixed_initial_state": False}
    #         env = Mech(**env_kwargs)
    #         # env = make_vec_env(Mech, n_envs=4, env_kwargs=env_kwargs) # TODO
    #         # pdb.set_trace()
            
    #         gnn_kwargs = {"hidden_channels":64, 
    #                       "out_channels":64, 
    #                       "normalize":False, 
    #                       "batch_normalization":False, 
    #                       "lin":True, 
    #                       "add_loop":False}
    #         dqn_arch = [64, 256, 1024, 4096]
    #         ppo_arch = [64, dict(vf=[32], pi=[256, 1024, 4096])]
    #         # env = make_vec_env(lambda: env, n_envs=1)
    #         policy_kwargs = dict(
    #             features_extractor_class=GNN,
    #             features_extractor_kwargs=gnn_kwargs,
    #             net_arch=dqn_arch,
    #         )
    #         model = CustomDQN(policy=CustomDQNPolicy,
    #                 env=env,
    #                 learning_rate=linear_schedule(1e-3),
    #                 buffer_size=100000,  # 1e6
    #                 learning_starts=500,
    #                 batch_size=512,
    #                 tau=1.0, # the soft update coefficient (“Polyak update”, between 0 and 1)
    #                 gamma=0.99,
    #                 train_freq=(1000, "step"),
    #                 gradient_steps=1,
    #                 replay_buffer_class=None,
    #                 replay_buffer_kwargs=None,
    #                 optimize_memory_usage=False,
    #                 target_update_interval=10000,
    #                 exploration_fraction=0.8, # percent of learning that includes exploration
    #                 exploration_initial_eps=1.0, # Initial random search
    #                 exploration_final_eps=0.05, # final stochasticity
    #                 max_grad_norm=10.,
    #                 tensorboard_log=tb_log_dir,
    #                 create_eval_env=False,
    #                 policy_kwargs=policy_kwargs,
    #                 verbose=0,
    #                 seed=None,
    #                 device="cuda:1",
    #                 _init_setup_model=True)
    #         # model = DQN(policy='MultiInputPolicy',
    #         #         env=env,
    #         #         learning_rate=linear_schedule(1e-4),
    #         #         buffer_size=100000,  # 1e6
    #         #         learning_starts=5000,
    #         #         batch_size=1024,
    #         #         tau=1.0,
    #         #         gamma=0.99,
    #         #         train_freq=4,
    #         #         gradient_steps=1,
    #         #         replay_buffer_class=None,
    #         #         replay_buffer_kwargs=None,
    #         #         optimize_memory_usage=False,
    #         #         target_update_interval=2500,
    #         #         exploration_fraction=0.8, # percent of learning that includes exploration
    #         #         exploration_initial_eps=1.0, # Initial random search
    #         #         exploration_final_eps=0.05, # final stochasticity
    #         #         max_grad_norm=10.,
    #         #         tensorboard_log=tb_log_dir,
    #         #         create_eval_env=False,
    #         #         policy_kwargs=policy_kwargs,
    #         #         verbose=0,
    #         #         seed=None,
    #         #         device="cuda:2",
    #         #         _init_setup_model=True)
    #         # model = PPO(policy=CustomActorCriticPolicy, 
    #         #             env=env, 
    #         #             learning_rate=1e-4, 
    #         #             n_steps=1000, 
    #         #             batch_size=500, 
    #         #             n_epochs=1, 
    #         #             gamma=0.99, 
    #         #             gae_lambda=0.95,
    #         #             clip_range=0.2, 
    #         #             clip_range_vf=None, 
    #         #             # normalize_advantage=True, 
    #         #             ent_coef=0.01,
    #         #             vf_coef=0.5, 
    #         #             max_grad_norm=0.5, 
    #         #             use_sde=False, 
    #         #             sde_sample_freq=-1,
    #         #             target_kl=None, 
    #         #             tensorboard_log=tb_log_dir, 
    #         #             create_eval_env=False, 
    #         #             policy_kwargs=policy_kwargs,
    #         #             verbose=1, 
    #         #             seed=None, 
    #         #             device="cuda:1", 
    #         #             _init_setup_model=True)
    #         # model = A2C(policy=CustomActorCriticPolicy,
    #         #     env=env,
    #         #     learning_rate=linear_schedule(1e-3),
    #         #     n_steps=5,
    #         #     gamma=0.99,
    #         #     gae_lambda=1.0,
    #         #     ent_coef=0.01,
    #         #     vf_coef=0.5,
    #         #     max_grad_norm=0.5,
    #         #     rms_prop_eps=1e-5,
    #         #     use_rms_prop=True,
    #         #     use_sde=False,
    #         #     sde_sample_freq=-1,
    #         #     normalize_advantage=False,
    #         #     tensorboard_log=tb_log_dir,
    #         #     create_eval_env=False,
    #         #     policy_kwargs=policy_kwargs,
    #         #     verbose=1,
    #         #     seed=0,
    #         #     device="cuda:0",
    #         #     _init_setup_model=True)
    #         # model.load("dqn_mech_v3")

    #         model.learn(20000)
    #         model_filename = uniquify(f"./{type(model).__name__}_mech_v4_{goal_filename}.zip")
    #         model.save(model_filename)
            
    #         if type(model).__name__ == "PPO": best_designs = env.get_attr('best_designs')
    #         else: best_designs = env.best_designs
            
    #         if best_designs:
    #             pickle.dump(best_designs, open(uniquify(f'best_designs_{goal_filename}.pkl'), 'wb'))

    # # model = DQN.load(f"./dqn_mech_v3_{goal_filename}")

    # # obs = env.reset()
    # # for _ in range(10):
    # #     action, _states = model.predict(obs, deterministic=False)
    # #     obs, reward, done, info = env.step(action)
    # #     env.render(show=True)
    # #     if done:
    # #         obs = env.reset()
    # # pdb.set_trace()

    # # env.init(node_positions=node_pos, edges=initial_edges, steps=50)
    # # pdb.set_trace()
    # # Step 2 check validity 
    # # while not env.is_valid():
    # #     # If not valid save data
    # #     node_pos_ind = np.random.choice(pos_ind, size=4, replace=False)
    # #     node_pos = pos[node_pos_ind,:]
    # #     env = Mech()
    # #     env.init(node_positions=node_pos, edges=initial_edges, steps=50)

    # # try:
    # # except Exception as e:
    # #     print(e)
    # #     pdb.set_trace()
    # # env.render(show=True)

    # # Tests
    # # Import random graph
    # # filename = 'saved_graphs/six_bar/six-bar1.pkl'
    # # g = pickle.load(open(filename, 'rb'))
    # # graph = Mech(node_positions=g.joints[:,:,0].T, edges=g.lam, steps=50)
    # # graph.update_fixed_paths(fixed_node_pos=np.array([[-1., 0.], graph.paths[3,:,0]-1]))
    # # graph.add_node(np.array([2., -2.]))
    # # graph.add_edge(graph.number_of_nodes()-1, 3)
    # # graph.add_edge(graph.number_of_nodes()-1, 2)
    # # graph.update_paths()
    # # graph.sample_workspace(5, 2)
    # # if graph.is_valid():
    # #     graph.plot_graph(plot_paths=True, filename="testing_mech.png")

