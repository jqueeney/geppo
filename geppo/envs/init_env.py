"""Interface to all environment files."""
from geppo.envs.wrappers.normalize_wrapper import NormEnv

def init_env(env_type,env_name,task_name,
    s_normalize,r_normalize,s_t,s_mean,s_var,r_t,r_mean,r_var):
    """Creates environment with NormEnv wrapper.
    
    Args:
        env_type (string): OpenAI Gym (gym) or DeepMind Control Suite (dmc)
        env_name (string): environment / domain name
        task_name (string): task name (dmc only)
        s_normalize (bool): if True, normalizes observations
        r_normalize (bool): if True, normalizes rewards
        s_t (int), s_mean, s_var (np.ndarrays): observation normalization stats
        r_t (int), r_mean, r_var (floats): reward normalization stats
    
    Returns:
        Normalized environment.
    """

    if env_type == 'gym':
        from geppo.envs.wrappers.gym_wrapper import make_gym_env
        env_raw = make_gym_env(env_name)
    elif env_type == 'dmc':
        from geppo.envs.wrappers.dmc_wrapper import make_dmc_env
        env_raw = make_dmc_env(env_name,task_name)
    else:
        raise ValueError('Only gym and dmc env_type supported')
    
    env = NormEnv(env_raw,s_normalize,r_normalize)
    env.set_rms(s_t,s_mean,s_var,r_t,r_mean,r_var)
    
    return env