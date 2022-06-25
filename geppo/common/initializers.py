"""Helper functions to initialize main objects used in training."""
import numpy as np
import tensorflow as tf
import random
import os
import gym

from geppo.common.actor import GaussianActor, SoftMaxActor
from geppo.common.critic import Critic

def init_seeds(seed,env=None):
    """Sets random seed."""
    seed = int(seed)
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_actor(env,actor_layers,actor_activations,actor_gain,actor_std_mult,
    actor_per_state_std,actor_weights):
    """Initializes actor."""
    if isinstance(env.action_space,gym.spaces.Box):
        actor = GaussianActor(env,actor_layers,actor_activations,actor_gain,
            actor_std_mult,actor_per_state_std)
    elif isinstance(env.action_space,gym.spaces.Discrete):
        actor = SoftMaxActor(env,actor_layers,actor_activations,actor_gain)
    else:
        raise TypeError('Only Gym Box and Discrete action spaces supported')

    if actor_weights is not None:
        actor.set_weights(actor_weights)
    
    return actor

def init_critic(env,critic_layers,critic_activations,critic_gain,
    critic_weights):
    """Initializes critic."""    
    critic = Critic(env,critic_layers,critic_activations,critic_gain)

    if critic_weights is not None:
        critic.set_weights(critic_weights)
    
    return critic