Models
==============

GCP-HOLO uses Stable Baselines 3 for reinforcement learning training, but includes some customization to make sure that it works for the Mech Gym environment. Stable Baselines 3 is a popular library for RL training that provides a set of pre-implemented algorithms, such as Proximal Policy Optimization (PPO) and Deep Q-Network (DQN).

GCP-HOLO customizes the Stable Baselines 3 algorithms to work specifically with the Mech Gym environment, which is a custom environment designed for path synthesis of linkage systems. The Mech Gym environment includes a specific action space to enhance the efficiency, the customization also include maksing invalid actions determined from the scaffold nodes and makes sure that each of the models is selecting actions non-deterministically.

A2C
-----------------

This is the custom A2C that GCP-HOLO uses.

.. automodule:: models.a2c
   :members:
   :undoc-members:
   :show-inheritance:

DQN
-----------------

This is the custom DQN that GCP-HOLO uses.

.. automodule:: models.dqn
   :members:
   :undoc-members:
   :show-inheritance:

GCN
------------------

This is the graph convolution policy network adopted from You et al. 

.. automodule:: models.gcpn
   :members:
   :undoc-members:
   :show-inheritance:

Random Search
----------------------------

This is the random search method for applying random actions for generating linkages.

.. automodule:: models.random_search
   :members:
   :undoc-members:
   :show-inheritance:


