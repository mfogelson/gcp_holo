import argparse
import multiprocessing
import os
import pdb
import pickle


import matplotlib.pyplot as plt
import numpy as np
from timebudget import timebudget
from itertools import repeat

import wandb

from linkage_gym.envs.Mech import Mech
from linkage_gym.utils.env_utils import normalize_curve, uniquify
from utils.utils import linear_schedule

from models.a2c import CustomActorCriticPolicy
from models.dqn import CustomDQN, CustomDQNPolicy
from models.gcpn import GNN
from models.random_search import random_search
from utils.utils import evaluate_policy, cmaes
# from stable_baselines3.common.evaluation import evaluate_policy


from stable_baselines3 import A2C, PPO  # , HER, PPO1, PPO2, TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from datetime import datetime
import time

import warnings
# warnings.simplefilter("ignore")
# def fxn():
#     warnings.warn("invalid", RuntimeWarning)

#! NEW 
import concurrent.futures
def random_search_wrapper(args):
            return random_search(*args)

@timebudget
def main(parameters):

    # parameters["body_constraints = [-3., 3., 0., 3.]
    # parameters["coupler_constraints = [-3., 3., -3., 0.]
    now = datetime.now().strftime("%m_%d_%Y_%HD:%M:%S")
    day = datetime.now().strftime("%m_%d_%Y") 
    
    ## Use WANDB for logging
    os.environ["WANDB_MODE"] = parameters["wandb_mode"]

    ## Learn model N times
    for trial in range(parameters["num_trials"]):
        ## Initialize WANBD for Logging
        run = wandb.init(project=parameters["wandb_project"], 
               entity='mfogelson',
               sync_tensorboard=True,
               )
        
        ## Adds all of the arguments as config variables
        wandb.config.update(parameters) 

        ## Log / eval / save location
        tb_log_dir = f"./logs/{parameters['goal_filename']}/{parameters['model']}/{day}/{run.id}"
        eval_dir = f"./evaluations/{parameters['goal_filename']}/{parameters['model']}/{day}/"
        save_dir = f"./trained/{parameters['goal_filename']}/{parameters['model']}/{day}/"
        design_dir = f"./designs/{parameters['goal_filename']}/{parameters['model']}/{day}/"
        
        if not os.path.isdir(tb_log_dir):
            os.makedirs(tb_log_dir, exist_ok=True)
            
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir, exist_ok=True)
            
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        if not os.path.isdir(design_dir):
            os.makedirs(design_dir, exist_ok=True)

        ## Load Goal information
        goal_curve = pickle.load(open(f'{parameters["goal_path"]}/{parameters["goal_filename"]}.pkl', 'rb')) # NOTE: sometimes ordering needs to be reversed add [:,::-1]
        parameters["sample_points"] = goal_curve.shape[1]
        print("Goal Curve Shape: ", goal_curve.shape)
        # idx = np.round(np.linspace(0, goal_curve.shape[1] - 1, parameters["sample_points"])).astype(int)
        goal = normalize_curve(goal_curve) #R@normalize_curve(goal_curve[:,::-1][:,idx])
        # goal[:, -1] = goal[:, 0]
    
        
        ## Initialize Gym ENV
        env_kwargs = {"max_nodes":parameters["max_nodes"], 
                        "bound":parameters["bound"], 
                        "resolution":parameters["resolution"], 
                        "sample_points":parameters["sample_points"],
                        "feature_points": parameters["feature_points"], 
                        "goal":goal, 
                        "normalize":parameters["normalize"], 
                        # "seed": parameters["seed+trial, 
                        "fixed_initial_state": parameters["fixed_initial_state"], 
                        "ordered_distance": parameters["ordered"], 
                        "constraints": [], #[parameters["body_constraints, parameters["coupler_constraints], 
                        "self_loops": parameters["use_self_loops"], 
                        "use_node_type": parameters["use_node_type"],
                        "min_nodes": 6, 
                        "debug": False}
        
        # If PPO A2C or DQN can use multiple envs while training
        if parameters["model"] in ['PPO', 'A2C', 'DQN']:
            env = make_vec_env(Mech, n_envs=parameters["n_envs"], env_kwargs=env_kwargs, seed=parameters["seed"], vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': 'fork'}) # NOTE: For faster training use SubProcVecEnv 
            
        else:
            env = []
            for i in range(parameters["n_envs"]):
                env_kwargs['seed'] = i+parameters["seed"]
                env.append(Mech(**env_kwargs))
                        
        ## GNN Args
        gnn_kwargs = {
                "max_nodes": parameters["max_nodes"],
                "num_features": 2*parameters["feature_points"]+int(parameters["use_node_type"]),
                "hidden_channels":64, 
                "out_channels":64, 
                "normalize":False, 
                "batch_normalization":parameters["batch_normalize"], 
                "lin":True, 
                "add_loop":False}
        
        ## Policy Architecture
        dqn_arch = [64, 256, 1024, 4096]
        ppo_arch = [64, dict(vf=[32], pi=[256, 1024, 4096])] ## NOTE: Not used
        if parameters["model"] == "DQN":
            policy_kwargs = dict(
                features_extractor_class=GNN,
                features_extractor_kwargs=gnn_kwargs,
                net_arch=dqn_arch,
            )
        else:
            policy_kwargs = dict(
                features_extractor_class=GNN,
                features_extractor_kwargs=gnn_kwargs,
                net_arch=dqn_arch,
            )
            
        ## Initialize Model
        if parameters["model"] == "DQN":
            assert parameters["save_freq"] > parameters["update_freq"]//parameters["n_envs"]
            
            model = CustomDQN(policy=CustomDQNPolicy,
                    env=env,
                    learning_rate=parameters["lr"],
                    buffer_size=parameters["buffer_size"],  # 1e6
                    learning_starts=1,
                    batch_size=parameters["batch_size"],
                    tau=1.0, # the soft update coefficient (“Polyak update”, between 0 and 1)
                    gamma=parameters["gamma"],
                    train_freq=(parameters["update_freq"]//parameters["n_envs"], "step"),
                    gradient_steps=parameters["opt_iter"],
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    # target_update_interval=500,
                    exploration_fraction=0.5, # percent of learning that includes exploration
                    exploration_initial_eps=1.0, # Initial random search
                    exploration_final_eps=0.2, # final stochasticity
                    max_grad_norm=10.,
                    tensorboard_log=tb_log_dir,
                    create_eval_env=False,
                    policy_kwargs=policy_kwargs,
                    verbose=parameters["verbose"],
                    seed=parameters["seed"]+trial,
                    device=parameters["cuda"],
                    _init_setup_model=True)
            
        elif parameters["model"] == "PPO":
            model = PPO(policy=CustomActorCriticPolicy, 
                        env=env, 
                        learning_rate= linear_schedule(parameters["lr"]), 
                        n_steps=parameters["update_freq"]//parameters["n_envs"], 
                        batch_size=parameters["batch_size"], 
                        n_epochs=parameters["opt_iter"], 
                        gamma=parameters["gamma"], 
                        gae_lambda=0.95,
                        clip_range=parameters["eps_clip"], 
                        clip_range_vf=None, 
                        # normalize_advantage=True, 
                        ent_coef=parameters["ent_coef"],
                        vf_coef=0.5, 
                        max_grad_norm=0.5, 
                        use_sde=False, 
                        sde_sample_freq=-1,
                        target_kl=None, 
                        tensorboard_log=tb_log_dir, 
                        create_eval_env=False, 
                        policy_kwargs=policy_kwargs,
                        verbose=parameters["verbose"], 
                        seed=parameters["seed"]+trial, 
                        device=parameters["cuda"], 
                        _init_setup_model=True)
            
        elif parameters["model"] == "A2C":
            model = A2C(policy=CustomActorCriticPolicy,
                        env=env,
                        learning_rate=parameters["lr"],
                        n_steps=parameters["update_freq"]//parameters["n_envs"],
                        gamma=parameters["gamma"],
                        gae_lambda=0.95,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        rms_prop_eps=1e-5,
                        use_rms_prop=True,
                        use_sde=False,
                        sde_sample_freq=-1,
                        normalize_advantage=False,
                        tensorboard_log=tb_log_dir,
                        create_eval_env=False,
                        policy_kwargs=policy_kwargs,
                        verbose=parameters["verbose"],
                        seed=parameters["seed"]+trial,
                        device=parameters["cuda"],
                        _init_setup_model=True)
             
        elif parameters["model"] in ["random", "Random"]:
            print("Starting random search ...")
            evaluation_rewards = []
            evaluation_designs = []
            # for _ in range(parameters["m_evals):
            output = []
            # for e in env:
            #     output.append(random_search(e, parameters["n_eval_episodes))
            st = time.time()
            max_processes = max(min(parameters["n_envs"], os.cpu_count() // 2), 1)
            # with multiprocessing.Pool(max_processes) as p:
            #     output = p.starmap(random_search, zip(env, repeat(parameters["n_eval_episodes"])))
          
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                output = executor.map(random_search_wrapper, zip(env, repeat(parameters["n_eval_episodes"])))
            
            print(f"Finished random search...{time.time()-st}")

            # import pdb 
            # pdb.set_trace()
                # p.close()
            
            best_designs = []
            rewards = []
            designs = []
            lengths = []
            for o in output:
                best_designs.append(o[0])
                rewards.append(o[1])
                designs.append(o[2])
                lengths.append(o[3])
            
            # best_designs = sum(best_designs, [])
            rewards = sum(rewards, [])
            designs = sum(designs, [])
            lengths = sum(lengths, [])
            
            wandb.log({
                    'eval/mean_episode_rew': np.mean((rewards)),
                    'eval/std_episode_rew': np.std((rewards)),
                    'eval/median_episode_rew': np.median((rewards)),
                    'eval/mean_episode_lengths': np.mean((lengths)),
                    'eval/std_episode_lengths': np.std((lengths)),
                    'eval/median_episode_lengths': np.median((lengths)),})
            evaluation_rewards.append(rewards)
            evaluation_designs.append(designs)

            best_dict = {}
            rewards = {}
            for o in best_designs:
                for k, v in o.items():
                    if k not in best_dict:
                        best_dict[k] = v
                        rewards[k] = v[-1]
                        continue
                    
                    if v[-1] > best_dict[k][-1]:
                        best_dict[k] = v
                        rewards[k] = v[-1]

            # node_info = [list(o.values()) for o in best_designs]
            node_info = best_dict.values() # sum(node_info, [])

                
            ## Log average reward
            if rewards: 
                for k, v in rewards.items():
                    wandb.log({
                            'Reward Best Designs': v, 'Number of Loops': k})
            
            best_cmaes = {}
            ## Plot Best Designs
            figs = []
            for node_positions, edges, _ in node_info:
                env_kwargs['node_positions'] = node_positions
                env_kwargs['edges'] = edges
                tmp_env = Mech(**env_kwargs)
                tmp_env.is_terminal = True
                
                reward = tmp_env._get_reward()
                # rewards.append(reward[0])
                
                number_of_cycles = tmp_env.number_of_cycles()
                # number_of_cycles_.append(number_of_cycles)
                
                # pdb.set_trace()

                fig = tmp_env.paper_plotting()
                plt.rcParams['font.size'] = 10
                fig.suptitle(f'Algo: {parameters["model"]} | ID: {run.id} |\n Reward: {np.round(reward[0], 3)} | Number Of Cycles: {number_of_cycles}')
                
                ## Log images
                wandb.log({'best_designs': wandb.Image(fig)})
                figs.append(fig)
                plt.close(fig)

                
                if parameters["cmaes"]: 
                    cma_env = cmaes(tmp_env, sigma=0.000055, tolfun=0.00001)
                    reward = cma_env._get_reward()
                    n = cma_env.number_of_nodes()
                    best_cmaes[number_of_cycles] = [cma_env.paths[:n,:,0], cma_env.get_edges(), cma_env.goal, cma_env.total_dist]
                    
                    fig = cma_env.paper_plotting()
                    plt.rcParams['font.size'] = 10
                    fig.suptitle(f'Algo: {parameters["model"]} | ID: {run.id} |\n Reward: {np.round(reward[0], 3)} | Number Of Cycles: {number_of_cycles}')
                    
                    ## Log images
                    wandb.log({'best_designs_cmaes': wandb.Image(fig)})
                    figs.append(fig)
                    plt.close(fig)

                        
            
            #TODO Toggle this 
            # pickle.dump([evaluation_rewards, evaluation_designs], open(f"evaluations/evaluation_{parameters['model']}_{parameters['goal_filename']}_{parameters['n_eval_episodes']*parameters['n_envs']}_{parameters['m_evals']}_{run.id}.pkl", 'wb'))
                
                
        
        train = not parameters["no_train"] ## TODO
        print(f"Training set to: {train}")
        ## Load old checkpoint
        if parameters["checkpoint"]:
            print('Loading Checkpoint ...')
            model = model.load(parameters["checkpoint"])
            # train = False
        
        ## Learn
        if parameters["model"] in ["DQN", 'A2C', 'PPO']:    

            if train:
                print("Starting Training...")
                
                # Save a checkpoint 
                callback = CheckpointCallback(save_freq=parameters["save_freq"]//parameters["n_envs"], save_path=save_dir, name_prefix=f'{now}_{parameters["model"]}_model_{parameters["goal_filename"]}')

                model.learn(parameters["steps"], log_interval=5, reset_num_timesteps=False, callback=callback) 
                
                print("Finished Training...")
                
                print("Saving Model...")
                model.save(save_dir + f'{now}_{parameters["model"]}_model_{parameters["goal_filename"]}_final.zip')
            # import pdb
            # pdb.set_trace()
            
            ## Evaluate Model
            evaluation_rewards = []
            evaluation_designs = []
            for i in range(parameters["m_evals"]):
                ## Initialize model seed
                model.set_random_seed(seed=i+parameters["seed"])
                
                print("Evaluating Model...")
                rewards, lengths, designs = evaluate_policy(model, env, n_eval_episodes=parameters["n_eval_episodes"], deterministic=False, render=False, return_episode_rewards=True) ## TODO: update
                
                wandb.log({
                    'eval/mean_episode_rew': np.mean((rewards)),
                    'eval/std_episode_rew': np.std((rewards)),
                    'eval/median_episode_rew': np.median((rewards)),
                    'eval/mean_episode_lengths': np.mean((lengths)),
                    'eval/std_episode_lengths': np.std((lengths)),
                    'eval/median_episode_lengths': np.median((lengths)),})
                evaluation_rewards.append(rewards)
                evaluation_designs.append(designs)
                print("Saving Evaluation Designs...")
                ## TODO toggle this feature
                # pickle.dump([evaluation_rewards, evaluation_designs], open(eval_dir + f"{now}_{parameters['model']}_{parameters['goal_filename']}_{parameters['n_eval_episodes']}_{parameters['m_evals']}_{run.id}.pkl", 'wb'))
                

                ## Extract Best Designs 
                if parameters["model"] in ["PPO", "A2C", "DQN"]: best_designs = env.get_attr('best_designs')

                if isinstance(best_designs, list):
                #     pdb.set_trace()
                    best_dict = {}
                    rewards = {}
                    for o in best_designs:
                        for k, v in o.items():
                            if k not in best_dict:
                                best_dict[k] = v
                                rewards[k] = v[-1]
                                continue
                            
                            if v[-1] > best_dict[k][-1]:
                                best_dict[k] = v
                                rewards[k] = v[-1]

                    # node_info = [list(o.values()) for o in best_designs]
                    node_info = best_dict.values() # sum(node_info, [])
                elif isinstance(best_designs, dict):
                    node_info = best_designs.values()
                
                best_cmaes = {}
                ## Plot Best Designs
                figs = []
                for node_positions, edges, _ in node_info:
                    env_kwargs['node_positions'] = node_positions
                    env_kwargs['edges'] = edges
                    tmp_env = Mech(**env_kwargs)
                    tmp_env.is_terminal = True
                    
                    reward = tmp_env._get_reward()
                    # rewards.append(reward[0])
                    
                    number_of_cycles = tmp_env.number_of_cycles()
                    # number_of_cycles_.append(number_of_cycles)
                    
                    # pdb.set_trace()

                    fig = tmp_env.paper_plotting()
                    plt.rcParams['font.size'] = 10
                    fig.suptitle(f'Algo: {parameters["model"]} | ID: {run.id} |\n Reward: {np.round(reward[0], 3)} | Number Of Cycles: {number_of_cycles}')
                    
                    ## Log images
                    wandb.log({'best_designs': wandb.Image(fig)})
                    figs.append(fig)
                    plt.close(fig)

                    
                    if parameters["cmaes"]: 
                        cma_env = cmaes(tmp_env, sigma=0.000055, tolfun=0.00001)
                        reward = cma_env._get_reward()
                        n = cma_env.number_of_nodes()
                        best_cmaes[number_of_cycles] = [cma_env.paths[:n,:,0], cma_env.get_edges(), cma_env.goal, cma_env.total_dist]
                        
                        fig = cma_env.paper_plotting()
                        plt.rcParams['font.size'] = 10
                        fig.suptitle(f'Algo: {parameters["model"]} | ID: {run.id} |\n Reward: {np.round(reward[0], 3)} | Number Of Cycles: {number_of_cycles}')
                        
                        ## Log images
                        wandb.log({'best_designs_cmaes': wandb.Image(fig)})
                        figs.append(fig)
                        plt.close(fig)

                            
                ## Log average reward
                if rewards: 
                    for k, v in rewards.items():
                        wandb.log({
                                'Reward Best Designs': v, 'Number of Loops': k})
                                # 'Standard Deviation Reward Best Designs': np.std(list(rewards.values())),
                                # 'Median Reward Best Designs': np.median(list(rewards.values())),})
        
                ## Save Designs
                if best_designs:
                    
                    pickle.dump(best_dict, open(uniquify(design_dir+f'best_designs_{run.id}.pkl'), 'wb'))
                    
                if best_cmaes:
                    pickle.dump(best_cmaes, open(uniquify(design_dir+f'best_designs_cmaes_{run.id}.pkl'), 'wb'))
                
        run.finish()

        return env_kwargs, best_dict, best_cmaes
        # time.sleep(1)

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add parameters positional/optional 
    # default_path = 'data/training_seeds.pkl'
    
    ## Env Args
    parser.add_argument('--max_nodes',           default=11,                      type=int,             help="Maximum number of revolute joints on linkage graph\n (default: %(default)s, type: %(type)s)")
    parser.add_argument('--resolution',          default=11,                      type=int,             help="Resolution of scaffold nodes (default: %(default)s, type: %(type)s)")
    parser.add_argument('--bound',               default=1.0,                     type=float,           help="Bound for linkage graph design [-bound, bound] (default: %(default)s, type: %(type)s)")
    parser.add_argument('--sample_points',       default=20,                      type=int,             help="Numbder of points to sample the trajectories of revolute joints (default: %(default)s, type: %(type)s)")
    parser.add_argument('--feature_points',      default=1,                       type=int,             help="Number of feature points for node vector used in GNN (default: %(default)s, type: %(type)s)")
    parser.add_argument('--goal_filename',       default='jansen_traj',           type=str,             help="Goal filename (default: %(default)s, type: %(type)s)")
    parser.add_argument('--goal_path',           default='data/other_curves',     type=str,             help="Path to goal file (default: %(default)s, type: %(type)s)")
    parser.add_argument('--use_self_loops',      default=False,                   action='store_true',  help="Add self-loops in adj matrix (default: %(default)s, type: %(type)s)")
    parser.add_argument('--normalize',           default=False,                   action='store_true',  help="Normalize trajectory for feature vector (default: %(default)s, type: %(type)s)")
    parser.add_argument('--use_node_type',       default=False,                   action='store_true',  help="Use node type id for feature vector (default: %(default)s, type: %(type)s)")
    parser.add_argument('--fixed_initial_state', default=False,                    action='store_true',  help="Use same initial design state for all training (default: %(default)s, type: %(type)s)")
    parser.add_argument('--seed',                default=123,                     type=int,             help="Random seed for numpy and gym (default: %(default)s, type: %(type)s)")
    parser.add_argument('--ordered',             default=True,                    action='store_true',  help="Get minimum ordered distance (default: %(default)s, type: %(type)s)")
    parser.add_argument('--body_constraints',    default=None,                    type=float, nargs='+',help="Constraint on Non-coupler revolute joints[xmin, xmax, ymin, ymax] (default: %(default)s, type: %(type)s)")
    parser.add_argument('--coupler_constraints', default=None,                    type=float, nargs='+',help="Constraint on Coupler joint [xmin, xmax, ymin, ymax] (default: %(default)s, type: %(type)s)")

    ## Feature Extractor Args
    parser.add_argument('--use_gnn',             default=True,          action='store_true',  help="Use GNN feature embedding (default: %(default)s, type: %(type)s)")
    parser.add_argument('--batch_normalize',     default=True,         action='store_true',   help="Use batch normalization in GNN (default: %(default)s, type: %(type)s)")
    
    ## Model Args
    parser.add_argument('--model',               default="PPO",         type=str,             help="Select which model type to use Models=[DQN, A2C, PPO, random] (default: %(default)s, type: %(type)s)")
    parser.add_argument('--n_envs',              default=1,             type=int,             help="Number of parallel environments to run (default: %(default)s, type: %(type)s)")
    parser.add_argument('--checkpoint',          default=None,          type=str,             help='Load a previous model checkpoint (default: %(default)s, type: %(type)s)')
    parser.add_argument('--update_freq',         default=1000,          type=int,             help="How often to update the model (default: %(default)s, type: %(type)s)")
    parser.add_argument('--opt_iter',            default=1,             type=int,             help="How many gradient steps per update (default: %(default)s, type: %(type)s)")
    parser.add_argument('--eps_clip',            default=0.2,           type=float,           help="PPO epsilon clipping (default: %(default)s, type: %(type)s)")
    parser.add_argument('--ent_coef',            default=0.01,           type=float,          help="PPO epsilon clipping (default: %(default)s, type: %(type)s)")

    parser.add_argument('--gamma',               default=0.99,          type=float,           help="Discount factor (default: %(default)s, type: %(type)s)")
    parser.add_argument('--lr',                  default=0.0001,       type=float,            help="Learning rate (default: %(default)s, type: %(type)s)")
    parser.add_argument('--batch_size',          default=1000,          type=int,             help="Batch Size for Dataloader (default: %(default)s, type: %(type)s)")
    parser.add_argument('--buffer_size',         default=1000000,          type=int,          help="Buffer size for DQN (default: %(default)s, type: %(type)s)")

    ## Training Args
    parser.add_argument('--steps',               default=50000,         type=int,             help='The number of steps to train (default: %(default)s, type: %(type)s)')
    parser.add_argument('--num_trials',          default=1,             type=int,             help="How many times to run a training of the model (default: %(default)s, type: %(type)s)")
    
    ## Evaluation Args
    parser.add_argument('--n_eval_episodes',     default=100,         type=int,               help='The number of epochs to evaluate the model (default: %(default)s, type: %(type)s)')
    parser.add_argument('--m_evals',             default=1,           type=int,               help="How many times to run the evaluation with varying seeds (default: %(default)s, type: %(type)s)")
    
    ## Other Args
    parser.add_argument('--log_freq',         default=1000,             type=int,             help="How often to log training values (default: %(default)s, type: %(type)s)")
    parser.add_argument('--save_freq',        default=10000,            type=int,             help="How often to save instances of model, buffer and render (default: %(default)s, type: %(type)s)")
    parser.add_argument('--wandb_mode',       default="online",         type=str,             help="use weights and biases to log information Modes=[online, offline, disabled] (default: %(default)s, type: %(type)s)")
    parser.add_argument('--wandb_project',    default="linkage_sb4",         type=str,        help="Set weights and biases project name (default: %(default)s, type: %(type)s)")

    parser.add_argument('--verbose',          default=0,                type=int,             help="verbose from sb3")
    parser.add_argument('--cuda',             default='cuda:1',         type=str,                help="Which GPU to use [cpu, cuda:0, cuda:1, cuda:2, cuda:3] (default: %(default)s, type: %(type)s)")
    parser.add_argument('--no_train',         default=False,         action='store_true',     help="If you don't want to train (default: %(default)s, type: %(type)s)")
    
    parser.add_argument('--cmaes',            default=False,         action='store_true',     help="Further optimize best designs found with CMA-ES node optimization (default: %(default)s, type: %(type)s)")

    
    # Parse the arguments
    args = parser.parse_args()
    
    print(list(vars(args).keys()))
    print(list(vars(args).values()))
    # pdb.set_trace()
    # Display Args
    print(args)
    
    
    # path = f"./trained/{parameters["goal_filename}/{parameters["model}/06_15_2022/"
    # parameters["checkpoint = path+os.listdir(path)[0]
    # print("CHECKPOINT LOCATION: ", parameters["checkpoint)
    
    # sys.settrace(trace)
    main(vars(args))
