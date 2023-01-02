import numpy as np 
import cma
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from copy import deepcopy
from timebudget import timebudget


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def func(input, env):
    """cma-es objective function

    Args:
        input (Nd.Array): new node positions
        env (gym.env): _description_

    Returns:
        float: reward
    """
    input = np.array(input).reshape(-1, 2)
    
    ## Update env
    env.paths[:input.shape[0], :, 0] = input
    env._initialize_paths()

    if env._get_reward()[0] == 0.0:
        return 1.0    
    
    return -np.nan_to_num(env._get_reward()[0], nan=-1e10)


def cmaes(env, tolfun=0.001, sigma=0.00001):
    """cma

    Args:
        env (gym.env): linkage_gym
        tolfun (float, optional): objective function tolerance. Defaults to 0.001.
        sigma (float, optional): distribution variance. Defaults to 0.00001.

    Returns:
        gym.env: optimized environment 
    """

    ## create a copy of the input
    opt_env = deepcopy(env)
    opt_env.init_args['edges'] = env.get_edges()
    
    ## run cma-es
    opts = {'bounds': [-1.0001, 1.0001], 'tolfun': tolfun}
    n = opt_env.number_of_nodes()
    es = cma.CMAEvolutionStrategy(opt_env.paths[:n,:,0].flatten(), sigma, inopts=opts)
    es.optimize(func, args=[opt_env]); 
    
    ## updated environment
    opt_env.paths[:n, :, 0] = es.result.xbest.reshape(-1,2)
    opt_env._initialize_paths()

    return opt_env

def evaluate_policy_simple(model, env, episodes):
    rewards = []
    designs = []
    lengths = []
    dist = []
    
    ## run for episodes
    for _ in range(episodes):
        env.reset()
        done = False
        l = 0
        while not done:
            actions, _ = model.predict(observations, state=None, deterministic=False)
            observations, rewards, dones, infos = env.step(actions)
            dis = env.total_dist
            l += 1
        
        ## Update lists
        dist.append(dis)
        rewards.append(reward)
        designs.append(info)
        lengths.append(l)
        
            
    return env.best_designs, rewards, designs, lengths, dist
## slightly modified by Mitch
@timebudget
def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_info = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None

    # episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, deterministic=deterministic)

        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        environment_stopping_criteria = (episode_counts < episode_count_targets)
        
        # valid = np.logical_and(environment_stopping_criteria, dones)
        tmp_rewards = []
        tmp_length = []
        tmp_info = []
        for esc, d, r, l, i in zip(environment_stopping_criteria, dones, current_rewards, current_lengths, infos):
            if esc and d:
                tmp_rewards.append(r)
                tmp_length.append(l)
                tmp_info.append(i)
        
        episode_rewards += tmp_rewards #list(current_rewards[valid])
        episode_lengths += tmp_length #list(current_lengths[valid])
        # # import pdb
        # # pdb.set_trace()
        episode_info += tmp_info #[i for i, v in zip(infos, valid) if v]
        
        current_lengths *= (1-dones)
        current_rewards *= (1-dones)
        
        
        episode_counts += dones
        
        
       
        # for i in range(n_envs):
        #     if episode_counts[i] < episode_count_targets[i]:

        #         # unpack values so that the callback can access the local variables
        #         reward = rewards[i]
        #         done = dones[i]
        #         info = infos[i]
        #         # episode_starts[i] = done

        #         if callback is not None:
        #             callback(locals(), globals())

        #         if dones[i]:
        #             episode_info.append(info)
        #             if is_monitor_wrapped:
        #                 # Atari wrapper can send a "done" signal when
        #                 # the agent loses a life, but it does not correspond
        #                 # to the true end of episode
        #                 if "episode" in info.keys():
        #                     # Do not trust "done" with episode endings.
        #                     # Monitor wrapper includes "episode" key in info if environment
        #                     # has been wrapped with it. Use those rewards instead.
        #                     episode_rewards.append(info["episode"]["r"])
        #                     episode_lengths.append(info["episode"]["l"])
        #                     # Only increment at the real end of an episode
        #                     episode_counts[i] += 1
        #             else:
        #                 episode_rewards.append(current_rewards[i])
        #                 episode_lengths.append(current_lengths[i])
                        
        #                 ## TODO: Add saving actual design too
        #                 # episode_paths.append(env)
        #                 episode_counts[i] += 1
        #             current_rewards[i] = 0
        #             current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_info
    return mean_reward, std_reward
