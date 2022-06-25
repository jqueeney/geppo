from collections import OrderedDict
import copy

import numpy as np
from dm_control import suite
from dm_env import specs
from dm_control.suite.wrappers import action_scale
from gym import spaces

def make_dmc_env(domain_name,task_name):
    """Creates DeepMind Control Suite task with actions scaled to [-1,1]."""
    return DMCWrapper(domain_name,task_name)

# From: https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/adapters/dm_control_adapter.py
def convert_dm_control_to_gym_space(dm_control_space):
    """Recursively convert dm_control_space into gym space.
    Note: Need to check the following cases of the input type, in the following
    order:
       (1) BoundedArraySpec
       (2) ArraySpec
       (3) OrderedDict.
    - Generally, dm_control observation_specs are OrderedDict with other spaces
      (e.g. ArraySpec) nested in it.
    - Generally, dm_control action_specs are of type `BoundedArraySpec`.
    To handle dm_control observation_specs as inputs, we check the following
    input types in order to enable recursive calling on each nested item.
    """
    if isinstance(dm_control_space, specs.BoundedArray):
        shape = dm_control_space.shape
        low = np.broadcast_to(dm_control_space.minimum, shape)
        high = np.broadcast_to(dm_control_space.maximum, shape)
        gym_box = spaces.Box(
            low=low,
            high=high,
            shape=None,
            dtype=dm_control_space.dtype)
        # Note: `gym.Box` doesn't allow both shape and min/max to be defined
        # at the same time. Thus we omit shape in the constructor and verify
        # that it's been implicitly set correctly.
        assert gym_box.shape == dm_control_space.shape, (
            (gym_box.shape, dm_control_space.shape))
        return gym_box
    elif isinstance(dm_control_space, specs.Array):
        if isinstance(dm_control_space, specs.BoundedArray):
            raise ValueError("The order of the if-statements matters.")
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(
                dm_control_space.shape
                if (len(dm_control_space.shape) == 1
                    or (len(dm_control_space.shape) == 3
                        and np.issubdtype(dm_control_space.dtype, np.integer)))
                else (int(np.prod(dm_control_space.shape)), )
            ),
            dtype=dm_control_space.dtype)
    elif isinstance(dm_control_space, OrderedDict):
        return spaces.Dict(OrderedDict([
            (key, convert_dm_control_to_gym_space(value))
            for key, value in dm_control_space.items()
        ]))
    else:
        raise ValueError(dm_control_space)

# Modified from: https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/adapters/dm_control_adapter.py
class DMCWrapper:
    """Wrapper to convert DeepMind Control Suite tasks to OpenAI Gym format."""

    def __init__(self,domain_name,task_name):
        """Initializes DeepMind Control Suite tasks.
        
        Args:
            domain_name (string): name of DeepMind Control Suite domain
            task_name (string): name of DeepMind Control Suite task
        """
        
        env = suite.load(domain_name=domain_name,task_name=task_name)
        assert isinstance(env.observation_spec(), OrderedDict)
        assert isinstance(env.action_spec(), specs.BoundedArray)
        
        env = action_scale.Wrapper(
            env,
            minimum=np.ones_like(env.action_spec().minimum)*-1,
            maximum=np.ones_like(env.action_spec().maximum)
        )
        np.testing.assert_equal(env.action_spec().minimum, -1)
        np.testing.assert_equal(env.action_spec().maximum, 1)
        self.env = env

        # Can remove parts of observation by excluding keys here
        self.observation_keys = tuple(env.observation_spec().keys())

        observation_space = convert_dm_control_to_gym_space(
            self.env.observation_spec())

        self.observation_space = type(observation_space)([
            (name, copy.deepcopy(space))
            for name, space in observation_space.spaces.items()
            if name in self.observation_keys
        ])

        self.action_space = convert_dm_control_to_gym_space(
            self.env.action_spec())

        if len(self.action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implementation.".format(self.action_space))
    
    def step(self,a):
        """Takes step in environment.

        Args:
            a (np.ndarray): action
        
        Returns:
            s (np.ndarray): flattened next state
            r (float): reward
            d (bool): done flag
            info (dict): dictionary with additional environment info
        """        
        time_step = self.env.step(a)
        r = time_step.reward or 0.0
        d = time_step.last()
        info = {
            key: value
            for key, value in time_step.observation.items()
            if key not in self.observation_keys
        }
        s = spaces.utils.flatten(self.observation_space,time_step.observation)
        return s, r, d, info

    def reset(self):
        """Resets environment and returns flattened initial state."""
        time_step = self.env.reset()
        s = spaces.utils.flatten(self.observation_space,time_step.observation)
        return s

    def seed(self,seed):
        self.env.task._random = np.random.RandomState(seed)
