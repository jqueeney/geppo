import gym

def make_gym_env(env_name):
    """Creates OpenAI Gym environment with actions scaled to [-1,1]."""
    env = gym.make(env_name)
    if isinstance(env.action_space,gym.spaces.Box):
        env = gym.wrappers.RescaleAction(env,-1.,1.)
    return env