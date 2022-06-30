from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
# from gcpn_gym import GNN
# from env.Mech_v4 import Mech
# from utils.env_utils import normalize_curve
# import pickle
# import pdb

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 5445,
        last_layer_dim_vf: int = 64,
        hidden_dim: int = 1024,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, last_layer_dim_pi),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, last_layer_dim_vf),
        )

    def forward(self, features: th.Tensor): # -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor): # -> th.Tensor:

        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor): # -> th.Tensor:

        return self.value_net(features)

# class ActorCriticPolicy(BasePolicy):
#     """
#     Policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param sde_net_arch: Network architecture for extracting features
#         when using gSDE. If None, the latent features from the policy will be used.
#         Pass an empty list to use the states as features.
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """

#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         sde_net_arch: Optional[List[int]] = None,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):

#         if optimizer_kwargs is None:
#             optimizer_kwargs = {}
#             # Small values to avoid NaN in Adam optimizer
#             if optimizer_class == th.optim.Adam:
#                 optimizer_kwargs["eps"] = 1e-5

#         super(ActorCriticPolicy, self).__init__(
#             observation_space,
#             action_space,
#             features_extractor_class,
#             features_extractor_kwargs,
#             optimizer_class=optimizer_class,
#             optimizer_kwargs=optimizer_kwargs,
#             squash_output=squash_output,
#         )

#         # Default network architecture, from stable-baselines
#         if net_arch is None:
#             if features_extractor_class == NatureCNN:
#                 net_arch = []
#             else:
#                 net_arch = [dict(pi=[64, 64], vf=[64, 64])]

#         self.net_arch = net_arch
#         self.activation_fn = activation_fn
#         self.ortho_init = ortho_init

#         self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs) ## GNN
#         self.features_dim = self.features_extractor.features_dim

#         self.normalize_images = normalize_images
#         self.log_std_init = log_std_init
#         dist_kwargs = None
#         # Keyword arguments for gSDE distribution
#         if use_sde:
#             dist_kwargs = {
#                 "full_std": full_std,
#                 "squash_output": squash_output,
#                 "use_expln": use_expln,
#                 "learn_features": False,
#             }

#         if sde_net_arch is not None:
#             warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

#         self.use_sde = use_sde
#         self.dist_kwargs = dist_kwargs

#         # Action distribution
#         self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

#         self._build(lr_schedule)

#     def _get_constructor_parameters(self) -> Dict[str, Any]:
#         data = super()._get_constructor_parameters()

#         default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

#         data.update(
#             dict(
#                 net_arch=self.net_arch,
#                 activation_fn=self.activation_fn,
#                 use_sde=self.use_sde,
#                 log_std_init=self.log_std_init,
#                 squash_output=default_none_kwargs["squash_output"],
#                 full_std=default_none_kwargs["full_std"],
#                 use_expln=default_none_kwargs["use_expln"],
#                 lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
#                 ortho_init=self.ortho_init,
#                 optimizer_class=self.optimizer_class,
#                 optimizer_kwargs=self.optimizer_kwargs,
#                 features_extractor_class=self.features_extractor_class,
#                 features_extractor_kwargs=self.features_extractor_kwargs,
#             )
#         )
#         return data

#     def reset_noise(self, n_envs: int = 1) -> None:
#         """
#         Sample new weights for the exploration matrix.

#         :param n_envs:
#         """
#         assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
#         self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

#     def _build_mlp_extractor(self) -> None:
#         """
#         Create the policy and value networks.
#         Part of the layers can be shared.
#         """
#         # Note: If net_arch is None and some features extractor is used,
#         #       net_arch here is an empty list and mlp_extractor does not
#         #       really contain any layers (acts like an identity module).
#         self.mlp_extractor = MlpExtractor(
#             self.features_dim,
#             net_arch=self.net_arch,
#             activation_fn=self.activation_fn,
#             device=self.device,
#         )

#     def _build(self, lr_schedule: Schedule) -> None:
#         """
#         Create the networks and the optimizer.

#         :param lr_schedule: Learning rate schedule
#             lr_schedule(1) is the initial learning rate
#         """
#         self._build_mlp_extractor()

#         latent_dim_pi = self.mlp_extractor.latent_dim_pi

#         if isinstance(self.action_dist, DiagGaussianDistribution):
#             self.action_net, self.log_std = self.action_dist.proba_distribution_net(
#                 latent_dim=latent_dim_pi, log_std_init=self.log_std_init
#             )
#         elif isinstance(self.action_dist, StateDependentNoiseDistribution):
#             self.action_net, self.log_std = self.action_dist.proba_distribution_net(
#                 latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
#             )
#         elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
#             self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
#         else:
#             raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

#         self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
#         # Init weights: use orthogonal initialization
#         # with small initial weight for the output
#         if self.ortho_init:
#             # TODO: check for features_extractor
#             # Values from stable-baselines.
#             # features_extractor/mlp values are
#             # originally from openai/baselines (default gains/init_scales).
#             module_gains = {
#                 self.features_extractor: np.sqrt(2),
#                 self.mlp_extractor: np.sqrt(2),
#                 self.action_net: 0.01,
#                 self.value_net: 1,
#             }
#             for module, gain in module_gains.items():
#                 module.apply(partial(self.init_weights, gain=gain))

#         # Setup optimizer with initial learning rate
#         self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

#     def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#         """
#         Forward pass in all the networks (actor and critic)

#         :param obs: Observation
#         :param deterministic: Whether to sample or use deterministic actions
#         :return: action, value and log probability of the action
#         """
#         # Preprocess the observation if needed
#         features = self.extract_features(obs)
#         latent_pi, latent_vf = self.mlp_extractor(features)
#         # Evaluate the values for the given observations
#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         actions = distribution.get_actions(deterministic=deterministic)
#         log_prob = distribution.log_prob(actions)
#         return actions, values, log_prob

#     def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
#         """
#         Retrieve action distribution given the latent codes.

#         :param latent_pi: Latent code for the actor
#         :return: Action distribution
#         """
#         mean_actions = self.action_net(latent_pi)

#         if isinstance(self.action_dist, DiagGaussianDistribution):
#             return self.action_dist.proba_distribution(mean_actions, self.log_std)
#         elif isinstance(self.action_dist, CategoricalDistribution):
#             # Here mean_actions are the logits before the softmax
#             return self.action_dist.proba_distribution(action_logits=mean_actions)
#         elif isinstance(self.action_dist, MultiCategoricalDistribution):
#             # Here mean_actions are the flattened logits
#             return self.action_dist.proba_distribution(action_logits=mean_actions)
#         elif isinstance(self.action_dist, BernoulliDistribution):
#             # Here mean_actions are the logits (before rounding to get the binary actions)
#             return self.action_dist.proba_distribution(action_logits=mean_actions)
#         elif isinstance(self.action_dist, StateDependentNoiseDistribution):
#             return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
#         else:
#             raise ValueError("Invalid action distribution")

#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         """
#         Get the action according to the policy for a given observation.

#         :param observation:
#         :param deterministic: Whether to use stochastic or deterministic actions
#         :return: Taken action according to the policy
#         """
#         return self.get_distribution(observation).get_actions(deterministic=deterministic)

    # def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     """
    #     Evaluate actions according to the current policy,
    #     given the observations.

    #     :param obs:
    #     :param actions:
    #     :return: estimated value, log likelihood of taking those actions
    #         and entropy of the action distribution.
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     latent_pi, latent_vf = self.mlp_extractor(features)
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     log_prob = distribution.log_prob(actions)
    #     values = self.value_net(latent_vf)
    #     return values, log_prob, distribution.entropy()



#     def predict_values(self, obs: th.Tensor) -> th.Tensor:
#         """
#         Get the estimated values according to the current policy given the observations.

#         :param obs:
#         :return: the estimated values.
#         """
#         features = self.extract_features(obs)
#         latent_vf = self.mlp_extractor.forward_critic(features)
#         return self.value_net(latent_vf)
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    # def _build_mlp_extractor(self) -> None:
    #     self.mlp_extractor = CustomNetwork(self.features_dim)

    def get_distribution(self, obs: th.Tensor):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, obs)

    def forward(self, obs: th.Tensor, deterministic: bool = False): # -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs) ## GNN 
        latent_pi, latent_vf = self.mlp_extractor(features) ## Latent Vectors from value and policy
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf) ## Value head 
        distribution = self._get_action_dist_from_latent(latent_pi, obs) ## Action Head
        actions = distribution.get_actions(deterministic=deterministic) ## Select Action
        log_prob = distribution.log_prob(actions) ## log_probs
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor): # -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs): # -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        ## Mitch Edit: Mask invalid actions 
        mask = obs[:, -self.action_space.n:].bool() #['action_mask'].bool()
        mean_actions = self.action_net(latent_pi).masked_fill_(~mask, -1e9) ## Action Head

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

# def main():
#     policy_kwargs = dict(
#                 features_extractor_class=GNN,
#                 # features_extractor_kwargs=dict(features_dim=128),
#             )

#     goal_filename = 'figure_eight_200'
#     goal = normalize_curve(pickle.load(open(f'{goal_filename}.pkl', 'rb')))

#     env = Mech(max_nodes=10, bound=1., resolution=11, sample_points=20, goal=goal, normalize=True)

#     model = PPO(CustomActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs)
#     import pdb
#     # pdb.set_trace()
#     model.learn(5000)

if __name__ == "__main__":
    main()