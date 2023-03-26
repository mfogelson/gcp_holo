.. GCP-HOLO documentation master file, created by
   sphinx-quickstart on Sat Mar 25 17:12:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GCP-HOLO's documentation!
====================================
.. admonition:: Important

   This is important information that you should pay attention to.
   
.. date:: 2023-03-26
The GCP-HOLO documentation describes a method for 1DOF path synthesis of high order linkage graphs. GCP-HOLO stands for Graph Convolution Policy for High Order Linkage Optimization. We define high order linkage graphs as more than 3 active loops, although GCP-HOLO can generate linkages with less number of loops, other methods will perform better. 

Path synthesis is a challenging task due to the non-linearities, combinatorial nature, and strict geometric constraints involved. GCP-HOLO iteratively grows a base linkage using Assur group 0DOF linkages. We are able to evaluate the new linkages quickly through symbolic kinematics described by Bacher et al. If you just niavely place a new joint and rigid bars to connect on to an existing 1DOF linkage, it is likely to be kinematically infeasible. To address this problem we introduce scaffold nodes which are potential initial joint locations. This gives the problem a large, but fixed action space to grow the linkage. The project allows you to train a Deep Reinforcement Learning (DRL) agent to learn a design heuristic for the path synthesis problem. This project supports 3 types of agents from the Stable Baselines3 library: PPO, A2C, and DQN, as well as a random search option as well. 


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   train
   mech
   linkage_gym.utils
   models
   utils
   gradio
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
