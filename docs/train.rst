Train
=====

This script is used to train a reinforcement learning model for the linkage gym environment.

Usage
-----

The script can be run with the following command:

.. code-block:: bash

   python train.py <options>

Options
-------

The following command-line options are available:

.. program:: train.py

.. option:: -h, --help

   Show the help message and exit.

.. option:: --max_nodes MAX_NODES

   Maximum number of revolute joints on linkage graph (default: 11, type: int)

.. option:: --resolution RESOLUTION

   Resolution of scaffold nodes (default: 11, type: int)

.. option:: --bound BOUND

   Bound for linkage graph design [-bound, bound] (default: 1.0, type: float)

.. option:: --sample_points SAMPLE_POINTS

   Number of points to sample the trajectories of revolute joints (default: 20, type: int)

.. option:: --feature_points FEATURE_POINTS

   Number of feature points for node vector used in GNN (default: 1, type: int)

.. option:: --goal_filename GOAL_FILENAME

   Goal filename (default: jansen_traj, type: str)

.. option:: --goal_path GOAL_PATH

   Path to goal file (default: data/other_curves, type: str)

.. option:: --use_self_loops

   Add self-loops in adj matrix (default: False, type: None)

.. option:: --normalize

   Normalize trajectory for feature vector (default: False, type: None)

.. option:: --use_node_type

   Use node type id for feature vector (default: False, type: None)

.. option:: --fixed_initial_state

   Use same initial design state for all training (default: True, type: None)

.. option:: --seed SEED

   Random seed for numpy and gym (default: 123, type: int)

.. option:: --ordered

   Get minimum ordered distance (default: True, type: None)

.. option:: --body_constraints BODY_CONSTRAINTS

   Constraint on Non-coupler revolute joints[xmin, xmax, ymin, ymax] (default: None, type: float)

.. option:: --coupler_constraints COUPLER_CONSTRAINTS

   Constraint on Coupler joint [xmin, xmax, ymin, ymax] (default: None, type: float)

.. option:: --use_gnn

   Use GNN feature embedding (default: True, type: None)

.. option:: --batch_normalize

   Use batch normalization in GNN (default: True, type: None)

.. option:: --model MODEL

   Select which model type to use Models=[DQN, A2C, PPO, random] (default: PPO, type: str)

.. option:: --n_envs N_ENVS

   Number of parallel environments to run (default: 1, type: int)

.. option:: --checkpoint CHECKPOINT

   Load a previous model checkpoint (default: None, type: str)

.. option:: --update_freq UPDATE_FREQ

   How often to update the model (default: 1000, type: int)

.. option:: --opt_iter OPT_ITER

   How many gradient steps per update (default: 1, type: int)

.. option:: --eps_clip EPS_CLIP

   PPO epsilon clipping (default: 0.2, type: float)

.. option:: --ent_coef ENT_COEF

   PPO epsilon clipping (default: 0.01, type: float)

.. option:: --gamma GAMMA

   Discount factor (default: 0.99, type: float)

.. option:: --lr LR

   Learning rate (default: 0.0001, type: float)

.. option:: --batch_size BATCH_SIZE

   Batch Size for Dataloader (default: 1000, type: int)

.. option:: --buffer_size BUFFER_SIZE

   Buffer size for DQN (default: 1000000, type: int)

.. option:: --steps STEPS

   The number of steps to train (default: 50000, type: int)

.. option:: --num_trials NUM_TRIALS

   How many times to run a training of the model (default: 1, type: int)

.. option:: --n_eval_episodes N_EVAL_EPISODES

   The number of epochs to evaluate the model (default: 100, type: int)

.. option:: --m_evals M_EVALS

   How many times to run the evaluation with varying seeds (default: 1, type: int)

.. option:: --log_freq LOG_FREQ

   How often to log training values (default: 1000, type: int)

.. option:: --save_freq SAVE_FREQ

   How often to save instances of model, buffer and render (default: 10000, type: int)

.. option:: --wandb_mode WANDB_MODE

   Use weights and biases to log information Modes=[online, offline, disabled] (default: online, type: str)

.. option:: --wandb_project WANDB_PROJECT

   Set weights and biases project name (default: linkage_sb4, type: str)

.. option:: --verbose VERBOSE

   Verbose from sb3 (default: 0)

.. option:: --cuda CUDA

   Which GPU to use [cpu, cuda:0, cuda:1, cuda:2, cuda:3] (default: cpu, type: str)

.. option:: --no_train

   If you don't want to train (default: False, type: None)

.. option:: --cmaes

   Further optimize best designs found with CMA-ES node optimization (default: False, type: None)

Example
-------

To train an A2C model for the Mech-v0 environment for 100 epochs with a batch size of 32 and a learning rate of 0.001, you can run:

.. code-block:: bash

   python train.py --model a2c --env Mech-v0 --epochs 100 --batch-size 32 --learning-rate 0.001

This will train the model and save the trained weights to the default output directory.
