import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit

from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
# from gcpn_gym import GNN
# from env.Mech_v4 import Mech
# from utils.env_utils import normalize_curve
# import pickle
# import pdb
from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy 
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class CustomQNetwork(QNetwork):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):

        super(CustomQNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim, 
            net_arch, 
            activation_fn,
            normalize_images,
        )

        # if net_arch is None:
        #     net_arch = [64, 64]

        # self.net_arch = net_arch
        # self.activation_fn = activation_fn
        # self.features_extractor = features_extractor
        # self.features_dim = features_dim
        # self.normalize_images = normalize_images
        # action_dim = self.action_space.n  # number of actions
        
        # q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        # self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor):
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        action_mask = obs[:, -self.action_space.n:].bool() #['action_mask'].bool()
        # import pdb
        # pdb.set_trace()
        return self.q_net(self.extract_features(obs)).masked_fill_(~action_mask, -1.0) #.softmax(1)

    # def _predict(self, observation: th.Tensor, deterministic: bool = True):
    #     q_values = self(observation)
    #     # Greedy action
    #     action = q_values.argmax(dim=1).reshape(-1)
        
    #     return action


class CustomDQNPolicy(DQNPolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CustomDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule, 
            net_arch, 
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        

        # if net_arch is None:
        #     if features_extractor_class == NatureCNN:
        #         net_arch = []
        #     else:
        #         net_arch = [64, 64]

        # self.net_arch = net_arch
        # self.activation_fn = activation_fn
        # self.normalize_images = normalize_images

        # self.net_args = {
        #     "observation_space": self.observation_space,
        #     "action_space": self.action_space,
        #     "net_arch": self.net_arch,
        #     "activation_fn": self.activation_fn,
        #     "normalize_images": normalize_images,
        # }

        # self.q_net, self.q_net_target = None, None
        # self._build(lr_schedule)
        

    # def _build(self, lr_schedule: Schedule) -> None:
    #     """
    #     Create the network and the optimizer.

    #     Put the target network into evaluation mode.

    #     :param lr_schedule: Learning rate schedule
    #         lr_schedule(1) is the initial learning rate
    #     """

    #     self.q_net = self.make_q_net()
    #     self.q_net_target = self.make_q_net()
    #     self.q_net_target.load_state_dict(self.q_net.state_dict())
    #     self.q_net_target.set_training_mode(False)

    #     # Setup optimizer with initial learning rate
    #     self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self):
        # Make sure we always have separate networks for features extractors etc
        # import pdb
        # pdb.set_trace()

        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return CustomQNetwork(**net_args).to(self.device)
    
    # def predict(
    #     self,
    #     observation: np.ndarray,
    #     state: Optional[Tuple[np.ndarray, ...]] = None,
    #     episode_start: Optional[np.ndarray] = None,
    #     deterministic: bool = False,
    # ):

    #     """
    #     Overrides the base_class predict function to include epsilon-greedy exploration.

    #     :param observation: the input observation
    #     :param state: The last states (can be None, used in recurrent policies)
    #     :param episode_start: The last masks (can be None, used in recurrent policies)
    #     :param deterministic: Whether or not to return deterministic actions.
    #     :return: the model's action and the next state
    #         (used in recurrent policies)
    #     """
    #     if not deterministic and np.random.rand() < self.exploration_rate:
    #         if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
    #             if isinstance(self.observation_space, gym.spaces.Dict):
    #                 n_batch = observation[list(observation.keys())[0]].shape[0]
    #             else:
    #                 n_batch = observation.shape[0]
    #             # valid_actions_mask = self._last_obs['action_mask'].flatten()
    #             # valid_actions_probs = valid_actions_mask/valid_actions_mask.sum()

    #             # all_actions = np.arange(self.action_space.n)
    #             all_actions = np.arange(self.action_space.n)
    #             action = np.array([np.random.choice(all_actions, 1, p=observation[i, -self.action_space.n:]/observation[i, -self.action_space.n:].sum()) for i in range(n_batch)])
    #             print('here')
    #             # action = np.array([np.random.choice(all_actions, 1, p=observation['action_mask'][i]/observation['action_mask'][i].sum()) for i in range(n_batch)])

    #         else:
    #             print('here')
    #             action = np.array(self.action_space.sample())

    #     else:
    #         action, state = self.policy.predict(observation, state, episode_start, deterministic)
            
    #     return action, state

    # def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
    #     return self._predict(obs, deterministic=deterministic)


    # def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
    #     return self.q_net._predict(obs, deterministic=deterministic)

    # def _get_constructor_parameters(self) -> Dict[str, Any]:
    #     data = super()._get_constructor_parameters()

    #     data.update(
    #         dict(
    #             net_arch=self.net_args["net_arch"],
    #             activation_fn=self.net_args["activation_fn"],
    #             lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
    #             optimizer_class=self.optimizer_class,
    #             optimizer_kwargs=self.optimizer_kwargs,
    #             features_extractor_class=self.features_extractor_class,
    #             features_extractor_kwargs=self.features_extractor_kwargs,
    #         )
    #     )
    #     return data

    # def set_training_mode(self, mode: bool) -> None:
    #     """
    #     Put the policy in either training or evaluation mode.

    #     This affects certain modules, such as batch normalisation and dropout.

    #     :param mode: if true, set to training mode, else set to evaluation mode
    #     """
    #     self.q_net.set_training_mode(mode)
    #     self.training = mode


class CustomDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(CustomDQN, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma, 
            train_freq, 
            gradient_steps, 
            replay_buffer_class, 
            replay_buffer_kwargs, 
            optimize_memory_usage, 
            target_update_interval, 
            exploration_fraction, 
            exploration_initial_eps, 
            exploration_final_eps, 
            max_grad_norm, 
            tensorboard_log, 
            create_eval_env, 
            policy_kwargs, 
            verbose, 
            seed, 
            device, 
            _init_setup_model
            # DQNPolicy,
            # learning_rate,
            # buffer_size,
            # learning_starts,
            # batch_size,
            # tau,
            # gamma,
            # train_freq,
            # gradient_steps,
            # # action_noise=None,  # No action noise
            # replay_buffer_class,
            # replay_buffer_kwargs,
            # policy_kwargs=policy_kwargs,
            # tensorboard_log=tensorboard_log,
            # verbose=verbose,
            # device=device,
            # create_eval_env=create_eval_env,
            # seed=seed,
            # # sde_support=False,
            # # optimize_memory_usage,
            # # supported_action_spaces=(gym.spaces.Discrete,),
            # # support_multi_env=True,
        )
        
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ):
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # for _ in range(n_envs):
            all_actions = np.arange(self.action_space.n)

            unscaled_action = np.array([np.random.choice(all_actions, 1, p=self._last_obs[i, -self.action_space.n:]/self._last_obs[i, -self.action_space.n:].sum()) for i in range(n_envs)])
            # unscaled_action = np.array([np.random.choice(all_actions, 1, p=self._last_obs['action_mask'][i]/self._last_obs['action_mask'][i].sum()) for i in range(n_envs)])
            # Warmup phase
            # valid_actions_mask = self._last_obs['action_mask'].flatten()
            # valid_actions_probs = valid_actions_mask/valid_actions_mask.sum()
            # # import pdb
            # # pdb.set_trace()
            # all_actions = np.arange(self.action_space.n)
            # import pdb
            # pdb.set_trace()
            # unscaled_action = np.random.choice(all_actions, size=n_envs, replace=False, p=valid_actions_probs)
            # unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
            
        return action, buffer_action
    
    def train(self, gradient_steps: int, batch_size: int = 100):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            # import pdb
            # pdb.set_trace()
            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            if np.isnan(loss.detach().cpu().numpy()).any():
                import pdb
                pdb.set_trace()
                raise ValueError
                
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):

        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # print("Correct Prediction Location")
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                # valid_actions_mask = self._last_obs['action_mask'].flatten()
                # valid_actions_probs = valid_actions_mask/valid_actions_mask.sum()

                # all_actions = np.arange(self.action_space.n)
                all_actions = np.arange(self.action_space.n)
                action = np.array([np.random.choice(all_actions, 1, p=observation[i, -self.action_space.n:]/observation[i, -self.action_space.n:].sum()) for i in range(n_batch)])
                # print('here')
                # action = np.array([np.random.choice(all_actions, 1, p=observation['action_mask'][i]/observation['action_mask'][i].sum()) for i in range(n_batch)])

            else:
                # print('here')
                action = np.array(self.action_space.sample())

        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
            
        return action, state

    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     train_freq: TrainFreq,
    #     replay_buffer: ReplayBuffer,
    #     action_noise: Optional[ActionNoise] = None,
    #     learning_starts: int = 0,
    #     log_interval: Optional[int] = None,
    # ):
    #     """
    #     Collect experiences and store them into a ``ReplayBuffer``.

    #     :param env: The training environment
    #     :param callback: Callback that will be called at each step
    #         (and at the beginning and end of the rollout)
    #     :param train_freq: How much experience to collect
    #         by doing rollouts of current policy.
    #         Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
    #         or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
    #         with ``<n>`` being an integer greater than 0.
    #     :param action_noise: Action noise that will be used for exploration
    #         Required for deterministic policy (e.g. TD3). This can also be used
    #         in addition to the stochastic policy for SAC.
    #     :param learning_starts: Number of steps before learning for the warm-up phase.
    #     :param replay_buffer:
    #     :param log_interval: Log data every ``log_interval`` episodes
    #     :return:
    #     """
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     episode_rewards, total_timesteps = [], []
    #     num_collected_steps, num_collected_episodes = 0, 0

    #     assert isinstance(env, VecEnv), "You must pass a VecEnv"
    #     assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
    #     assert train_freq.frequency > 0, "Should at least collect one step or episode."

    #     if self.use_sde:
    #         self.actor.reset_noise()

    #     callback.on_rollout_start()
    #     continue_training = True

    #     while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
    #         done = False
    #         episode_reward, episode_timesteps = 0.0, 0

    #         while not done:

    #             if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
    #                 # Sample a new noise matrix
    #                 self.actor.reset_noise()

    #             # Select action randomly or according to policy
    #             action, buffer_action = self._sample_action(learning_starts, action_noise)

    #             # Rescale and perform action
    #             new_obs, reward, done, infos = env.step(action)

    #             self.num_timesteps += 1
    #             episode_timesteps += 1
    #             num_collected_steps += 1

    #             # Give access to local variables
    #             callback.update_locals(locals())
    #             # Only stop training if return value is False, not when it is None.
    #             if callback.on_step() is False:
    #                 return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

    #             episode_reward += reward

    #             # Retrieve reward and episode length if using Monitor wrapper
    #             self._update_info_buffer(infos, done)

    #             # Store data in replay buffer (normalized action and unnormalized observation)
    #             self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

    #             self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

    #             # For DQN, check if the target network should be updated
    #             # and update the exploration schedule
    #             # For SAC/TD3, the update is done as the same time as the gradient update
    #             # see https://github.com/hill-a/stable-baselines/issues/900
    #             self._on_step()

    #             if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
    #                 break

    #         if done:
    #             num_collected_episodes += 1
    #             self._episode_num += 1
    #             episode_rewards.append(episode_reward)
    #             total_timesteps.append(episode_timesteps)

    #             if action_noise is not None:
    #                 action_noise.reset()

    #             # Log training infos
    #             if log_interval is not None and self._episode_num % log_interval == 0:
    #                 self._dump_logs()

    #     mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

    #     callback.on_rollout_end()

    #     return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
