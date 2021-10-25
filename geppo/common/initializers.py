"""Helper functions to initialize main objects used in training."""
import numpy as np
import tensorflow as tf
import random
import os
import gym
import pickle

from geppo.common.env_wrapper import NormEnv
from geppo.common.actor import GaussianActor
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

def init_env(env_name,s_normalize,r_normalize,
    s_t,s_mean,s_var,r_t,r_mean,r_var):
    """Creates environment with NormEnv wrapper."""
    env_raw = gym.make(env_name)
    env = NormEnv(env_raw,s_normalize,r_normalize)
    env.set_rms(s_t,s_mean,s_var,r_t,r_mean,r_var)
    
    return env

def init_actor(env,layers,activations,gain,std_mult,actor_weights):
    """Initializes actor."""
    actor = GaussianActor(env,layers,activations,gain,std_mult)

    if actor_weights is not None:
        actor.set_weights(actor_weights)
    
    return actor

def init_critic(env,layers,activations,gain,critic_weights):
    """Initializes critic."""    
    critic = Critic(env,layers,activations,gain)

    if critic_weights is not None:
        critic.set_weights(critic_weights)
    
    return critic

def import_params(import_path,import_file,import_idx):
    """Imports parameter info from previous simulation.
    
    Args:
        import_path (str): path where previous simulation is saved
        import_file (str): file name of previous simulation
        import_idx (int): index of simulation to import
    
    Returns:
        Dictionary of parameter info from previous simulation.
    """
    import_filefull = os.path.join(import_path,import_file)
    with open(import_filefull,'rb') as f:
        import_data = pickle.load(f)

    if isinstance(import_data,list):
        assert import_idx < len(import_data), 'import_idx too large'
        import_final = import_data[import_idx]['final']
        import_params = import_data[import_idx]['param']
    else:
        raise TypeError('imported data not a list')
    
    imported = dict()
    
    # Environment info
    imported['env_name'] = import_params['env_name']
    imported['s_normalize'] = import_params['s_normalize']
    imported['r_normalize'] = import_params['r_normalize']
    imported['s_t'] = import_final['s_t']
    imported['s_mean'] = import_final['s_mean']
    imported['s_var'] = import_final['s_var']
    imported['r_t'] = import_final['r_t']
    imported['r_mean'] = import_final['r_mean']
    imported['r_var'] = import_final['r_var']

    # Actor info
    imported['actor_layers'] = import_params['actor_layers']
    imported['actor_activations'] = import_params['actor_activations']
    imported['actor_gain'] = import_params['actor_gain']
    imported['actor_std_mult'] = import_params['actor_std_mult']
    imported['actor_weights'] = import_final['actor_weights']

    # Critic info
    imported['critic_layers'] = import_params['critic_layers']
    imported['critic_activations'] = import_params['critic_activations']
    imported['critic_gain'] = import_params['critic_gain']
    imported['critic_weights'] = import_final['critic_weights']

    return imported