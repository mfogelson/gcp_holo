Mech Gym Environment
====================

:class:`Mech` follows the `gym.Env` interface.


The Mech Gym environment is a custom environment designed for path synthesis of linkage systems, and it keeps track of various aspects of the linkage throughout the generation process. This includes the revolute joint locations for all time steps, the edges between revolute joints, the set of scaffold nodes, all action combinations, the goal trajectory, the distance of the coupler to the goal, the reward, and many other things.

Despite the complexity of the Mech Gym environment, it provides all the main functionality of an OpenAI Gym Env, with additional methods that can be useful. The environment should be designed as several classes, but during development, it was easier to keep everything in one class for simplicity. I hope to refactor this in the future.

Usage
-----

Create an instance of the Mechanism environment:

.. code-block:: bash

   Mech(**env_kwargs)

where ``env_kwargs`` is a dictionary of keyword arguments to pass to the environment. The following keyword arguments are supported:



Methods
-----------------------------
.. automodule:: linkage_gym.envs.Mech
   :members:
   :undoc-members:
   :show-inheritance:
