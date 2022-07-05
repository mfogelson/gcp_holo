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
from utils.utils import evaluate_policy
# from stable_baselines3.common.evaluation import evaluate_policy


from stable_baselines3 import A2C, PPO  # , HER, PPO1, PPO2, TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from datetime import datetime

import warnings
# warnings.simplefilter("ignore")
# def fxn():
#     warnings.warn("invalid", RuntimeWarning)

@timebudget
def main(args):
    # pdb.set_trace()
    # args.body_constraints = [-3., 3., 0., 3.]
    # args.coupler_constraints = [-3., 3., -3., 0.]
    now = datetime.now().strftime("%m_%d_%Y_%HD:%M:%S")
    day = datetime.now().strftime("%m_%d_%Y") 
    
    ## Use WANDB for logging
    os.environ["WANDB_MODE"] = args.wandb_mode

    ## Learn model N times
    for trial in range(args.num_trials):
        ## Initialize WANBD for Logging
        run = wandb.init(project=args.wandb_project, 
               entity='mfogelson',
               sync_tensorboard=True,
               )
        
        ## Adds all of the arguments as config variables
        wandb.config.update(args) 

        ## Log / eval / save location
        tb_log_dir = f"./logs/{args.goal_filename}/{args.model}/{day}/{run.id}"
        eval_dir = f"./evaluations/{args.goal_filename}/{args.model}/{day}/"
        save_dir = f"./trained/{args.goal_filename}/{args.model}/{day}/"
        design_dir = f"./designs/{args.goal_filename}/{args.model}/{day}/"
        
        if not os.path.isdir(tb_log_dir):
            os.makedirs(tb_log_dir, exist_ok=True)
            
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir, exist_ok=True)
            
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        if not os.path.isdir(design_dir):
            os.makedirs(design_dir, exist_ok=True)

        ## Load Goal information
        goal_curve = pickle.load(open(f'{args.goal_path}/{args.goal_filename}.pkl', 'rb')) # NOTE: sometimes ordering needs to be reversed add [:,::-1]
            
        idx = np.round(np.linspace(0, goal_curve.shape[1] - 1, args.sample_points)).astype(int)
        goal = normalize_curve(goal_curve[:,idx]) #R@normalize_curve(goal_curve[:,::-1][:,idx])
        goal[:, -1] = goal[:, 0]
        
        ## Initialize Gym ENV
        env_kwargs = {"max_nodes":args.max_nodes, 
                        "bound":args.bound, 
                        "resolution":args.resolution, 
                        "sample_points":args.sample_points,
                        "feature_points": args.feature_points, 
                        "goal":goal, 
                        "normalize":args.normalize, 
                        # "seed": args.seed+trial, 
                        "fixed_initial_state": args.fixed_initial_state, 
                        "ordered_distance": args.ordered, 
                        "constraints": [], #[args.body_constraints, args.coupler_constraints], 
                        "self_loops": args.use_self_loops, 
                        "use_node_type": args.use_node_type,}
        
        # If PPO A2C or DQN can use multiple envs while training
        if args.model in ['PPO', 'A2C', 'DQN']:
            env = make_vec_env(Mech, n_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seed) # NOTE: For faster training use SubProcVecEnv 
        else:
            env = []
            for i in range(args.n_envs):
                env_kwargs['seed'] = i+args.seed
                env.append(Mech(**env_kwargs))
                        
        ## GNN Args
        gnn_kwargs = {
                "max_nodes": args.max_nodes,
                "num_features": 2*args.feature_points+int(args.use_node_type),
                "hidden_channels":64, 
                "out_channels":64, 
                "normalize":False, 
                "batch_normalization":args.batch_normalize, 
                "lin":True, 
                "add_loop":False}
        
        ## Policy Architecture
        dqn_arch = [64, 256, 1024, 4096]
        ppo_arch = [64, dict(vf=[32], pi=[256, 1024, 4096])] ## NOTE: Not used
        if args.model == "DQN":
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
        if args.model == "DQN":
            assert args.save_freq > args.update_freq//args.n_envs
            
            model = CustomDQN(policy=CustomDQNPolicy,
                    env=env,
                    learning_rate=args.lr,
                    buffer_size=args.buffer_size,  # 1e6
                    learning_starts=1,
                    batch_size=args.batch_size,
                    tau=1.0, # the soft update coefficient (“Polyak update”, between 0 and 1)
                    gamma=args.gamma,
                    train_freq=(args.update_freq//args.n_envs, "step"),
                    gradient_steps=args.opt_iter,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    # target_update_interval=500,
                    exploration_fraction=0.8, # percent of learning that includes exploration
                    exploration_initial_eps=1.0, # Initial random search
                    exploration_final_eps=0.2, # final stochasticity
                    max_grad_norm=10.,
                    tensorboard_log=tb_log_dir,
                    create_eval_env=False,
                    policy_kwargs=policy_kwargs,
                    verbose=args.verbose,
                    seed=args.seed+trial,
                    device=args.cuda,
                    _init_setup_model=True)
            
        elif args.model == "PPO":
            model = PPO(policy=CustomActorCriticPolicy, 
                        env=env, 
                        learning_rate=linear_schedule(args.lr), 
                        n_steps=args.update_freq//args.n_envs, 
                        batch_size=args.batch_size, 
                        n_epochs=args.opt_iter, 
                        gamma=args.gamma, 
                        gae_lambda=0.95,
                        clip_range=args.eps_clip, 
                        clip_range_vf=None, 
                        # normalize_advantage=True, 
                        ent_coef=args.ent_coef,
                        vf_coef=0.5, 
                        max_grad_norm=0.5, 
                        use_sde=False, 
                        sde_sample_freq=-1,
                        target_kl=None, 
                        tensorboard_log=tb_log_dir, 
                        create_eval_env=False, 
                        policy_kwargs=policy_kwargs,
                        verbose=args.verbose, 
                        seed=args.seed+trial, 
                        device=args.cuda, 
                        _init_setup_model=True)
            
        elif args.model == "A2C":
            model = A2C(policy=CustomActorCriticPolicy,
                        env=env,
                        learning_rate=args.lr,
                        n_steps=args.update_freq//args.n_envs,
                        gamma=args.gamma,
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
                        verbose=args.verbose,
                        seed=args.seed+trial,
                        device=args.cuda,
                        _init_setup_model=True)
             
        elif args.model == "random":
            print("Starting random search ...")
            evaluation_rewards = []
            evaluation_designs = []
            # for _ in range(args.m_evals):
            output = []
            # for e in env:
            #     output.append(random_search(e, args.n_eval_episodes))
            with multiprocessing.Pool(max(args.n_envs, os.cpu_count()//2)) as p:
                output = p.starmap(random_search, zip(env, repeat(args.n_eval_episodes)))
            
            print("Finished random search...")
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
            
            pickle.dump([evaluation_rewards, evaluation_designs], open(f"evaluations/evaluation_{args.model}_{args.goal_filename}_{args.n_eval_episodes*args.n_envs}_{args.m_evals}_{run.id}.pkl", 'wb'))
                
                
        # elif args.model == "mcts":
        #     mcts_search(env, timesteps=args.steps)
            
        # elif args.model == "SA":
        #     simulated_annealing(env, args.steps)
        
        
        train = not args.no_train ## TODO
        print(f"Training set to: {train}")
        ## Load old checkpoint
        if args.checkpoint:
            print('Loading Checkpoint ...')
            model = model.load(args.checkpoint)
            # train = False
        
        ## Learn
        if args.model in ["DQN", 'A2C', 'PPO']:    

            if train:
                print("Starting Training...")
                # Save a checkpoint 
                callback = CheckpointCallback(save_freq=args.save_freq//args.n_envs, save_path=save_dir, name_prefix=f'{now}_{args.model}_model_{args.goal_filename}')
                    
                model.learn(args.steps, log_interval=5, reset_num_timesteps=False, callback=callback) 
                
                print("Finished Training...")
                 
                print("Saving Model...")
                model.save(save_dir + f'{now}_{args.model}_model_{args.goal_filename}_final.zip')
            
            
            ## Evaluate Model
            evaluation_rewards = []
            evaluation_designs = []
            for i in range(args.m_evals):
                ## Initialize model seed
                model.set_random_seed(seed=i+args.seed)
                
                print("Evaluating Model...")
                rewards, lengths, designs = evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes, deterministic=False, render=False, return_episode_rewards=True) ## TODO: update
                
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
            pickle.dump([evaluation_rewards, evaluation_designs], open(eval_dir + f"{now}_{args.model}_{args.goal_filename}_{args.n_eval_episodes}_{args.m_evals}_{run.id}.pkl", 'wb'))
                

        ## Extract Best Designs 
        if args.model in ["PPO", "A2C", "DQN"]: best_designs = env.get_attr('best_designs')

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
        
        ## Plot Best Designs
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
            fig.suptitle(f'Algo: {args.model} | ID: {run.id} |\n Reward: {np.round(reward[0], 3)} | Number Of Cycles: {number_of_cycles}')
            
            ## Log images
            wandb.log({'best_designs': wandb.Image(fig)})
            
            plt.close(fig)
            # fig.savefig('test_fig.png')
        
        ## Log average reward
        if rewards: 
            for k, v in rewards.items():
                wandb.log({
                        'Reward Best Designs': v, 'Number of Nodes': k})
                        # 'Standard Deviation Reward Best Designs': np.std(list(rewards.values())),
                        # 'Median Reward Best Designs': np.median(list(rewards.values())),})
        
        ## Save Designs
        if best_designs:
            
            pickle.dump(best_dict, open(uniquify(design_dir+f'best_designs_{run.id}.pkl'), 'wb'))
        
        run.finish()
        # time.sleep(1)

def trace(frame, event, arg):
    # print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add parameters positional/optional 
    # default_path = 'data/training_seeds.pkl'
    
    ## Env Args
    parser.add_argument('--max_nodes',           default=11,                      type=int,             help="number of design steps for each agent")
    parser.add_argument('--resolution',          default=11,                      type=int,             help="design resolution")
    parser.add_argument('--bound',               default=1.0,                     type=float,           help="[x,y] value range for revolute joints")
    parser.add_argument('--sample_points',       default=20,                      type=int,             help="number of sample points from FK")
    parser.add_argument('--feature_points',      default=1,                       type=int,             help="number of feature poinys for GNN")
    parser.add_argument('--goal_filename',       default='jansen_traj',           type=str,             help="Goal filename")
    parser.add_argument('--goal_path',           default='data/other_curves',     type=str,             help="path to goal file")
    parser.add_argument('--use_self_loops',      default=False,                   action='store_true',  help="Add self loops in adj matrix")
    parser.add_argument('--normalize',           default=True,                    action='store_true',  help="normalize trajectory for feature vector")
    parser.add_argument('--use_node_type',       default=False,                   action='store_true',  help="use node type id for feature vector")
    parser.add_argument('--fixed_initial_state', default=True,                    action='store_true',  help="use same initial state for all training")
    parser.add_argument('--seed',                default=123,             type=int,             help="Random seed for numpy and gym")
    parser.add_argument('--ordered',             default=True,          action='store_true',  help="Get minimum ordered distance")
    parser.add_argument('--body_constraints',    default=None,            type=float, nargs='+',  help="Non-coupler [xmin, xmax, ymin, ymax]")
    parser.add_argument('--coupler_constraints', default=None,            type=float, nargs='+',  help="coupler [xmin, xmax, ymin, ymax]")

    ## Feature Extractor Args
    parser.add_argument('--use_gnn',             default=True,          action='store_true',  help="use GNN embedding")
    parser.add_argument('--batch_normalize',     default=True,         action='store_true',  help="use batch norm")
    
    ## Model Args
    parser.add_argument('--model',               default="PPO",         type=str,             help="which model type to use Models=[DQN, A2C, PPO]")
    parser.add_argument('--n_envs',              default=1,             type=int,             help="number of parallel environments to run NOTE: only valid for A2C or PPO")
    parser.add_argument('--checkpoint',          default=None,          type=str,             help='A previous model checkpoint')
    parser.add_argument('--update_freq',         default=1000,          type=int,             help="how often to update the model via PPO")
    parser.add_argument('--opt_iter',            default=1,             type=int,             help="how many gradient steps per update")
    parser.add_argument('--eps_clip',            default=0.2,           type=float,           help="PPO epsilon clipping")
    parser.add_argument('--ent_coef',            default=0.01,           type=float,           help="PPO epsilon clipping")

    parser.add_argument('--gamma',               default=0.99,          type=float,           help="Discount factor")
    parser.add_argument('--lr',                  default=0.0001,       type=float,           help="Learning rate")
    parser.add_argument('--batch_size',          default=1000,          type=int,             help="Batch Size for Dataloader")
    parser.add_argument('--buffer_size',         default=1000000,          type=int,             help="Batch Size for Dataloader")

    ## Training Args
    parser.add_argument('--steps',               default=50000,         type=int,             help='The number of epochs to train')
    parser.add_argument('--num_trials',          default=1,             type=int,             help="How many times to run a training of the model")
    
    ## Evaluation Args
    parser.add_argument('--n_eval_episodes',     default=100,         type=int,             help='The number of epochs to train')
    parser.add_argument('--m_evals',             default=1,             type=int,             help="How many times to run a training of the model")
    
    ## Other Args
    parser.add_argument('--log_freq',         default=1000,             type=int,             help="how often to log to document")
    parser.add_argument('--save_freq',        default=10000,            type=int,             help="how often to save instances of model, buffer and render")
    parser.add_argument('--wandb_mode',       default="online",         type=str,             help="use weights and biases to log information Modes=[online, offline, disabled]")
    parser.add_argument('--wandb_project',    default="linkage_sb4",         type=str,             help="use weights and biases to log information Modes=[online, offline, disabled]")

    parser.add_argument('--verbose',          default=0,                type=int,             help="verbose from sb3")
    parser.add_argument('--cuda',             default='cpu',         type=str,             help="Which GPU to use [0, 1, 2, 3]")
    parser.add_argument('--no_train',         default=False,         action='store_true',             help="If you don't want to train")

    
    # Parse the arguments
    args = parser.parse_args()
    # pdb.set_trace()
    # Display Args
    print(args)
    
    
    # path = f"./trained/{args.goal_filename}/{args.model}/06_15_2022/"
    # args.checkpoint = path+os.listdir(path)[0]
    # print("CHECKPOINT LOCATION: ", args.checkpoint)
    
    # sys.settrace(trace)
    main(args)
