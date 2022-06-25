import numpy as np
import gym

class RunningNormalizer:
    """Class that tracks running statistics to use for normalization."""

    def __init__(self,dim):
        """"Initializes RunningNormalizer.

        Args:
            dim (int): dimension of statistic
        """
        self.dim = dim
        self.t_last = 0

        if dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.std = 1.0
        else:            
            self.mean = np.zeros(dim,dtype=np.float32)
            self.var = np.zeros(dim,dtype=np.float32)
            self.std = np.ones(dim,dtype=np.float32)
    
    def normalize(self,data,center=True,clip=10.):
        """Normalizes input data.
        
        Args:
            data (np.ndarray): data to normalize
            center (bool): if True, center data using running mean
            clip (float): value for clipping normalized data

        Returns:
            Normalized data.
        """
        if center:
            stand = (data - self.mean) / np.maximum(self.std,1e-8)
        else:
            stand = data / np.maximum(self.std,1e-8)
        
        return np.clip(stand,-clip,clip)

    def denormalize(self,data_norm,center=True):
        """Denormalizes input data.

        Args:
            data_norm (np.ndarray): normalized data
            center (bool): whether normalized data has been centered
        
        Returns:
            Denormalized data.
        """
        if center:
            data = data_norm * np.maximum(self.std,1e-8) + self.mean
        else:
            data = data_norm * np.maximum(self.std,1e-8)
        
        return data

    def update(self,data):
        """Updates statistics based on batch of data."""
        t_batch = data.shape[0]
        M_batch = data.mean(axis=0)
        S_batch = np.sum(np.square(data - M_batch),axis=0)

        t = t_batch + self.t_last

        self.var = ((S_batch + self.var * np.maximum(1,self.t_last-1)  
            + (t_batch / t) * self.t_last * np.square(M_batch-self.mean)) 
            / np.maximum(1,t-1))

        self.mean = (t_batch * M_batch + self.t_last * self.mean) / t

        self.mean = self.mean.astype('float32')
        self.var = self.var.astype('float32')

        if t==1:
            self.std = np.abs(self.mean)
        else:
            self.std = np.sqrt(self.var)

        self.t_last = t
    
    def reset(self):
        """Resets statistics."""
        self.t_last = 0

        if self.dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.std = 1.0
        else:            
            self.mean = np.zeros(self.dim,dtype=np.float32)
            self.var = np.zeros(self.dim,dtype=np.float32)
            self.std = np.ones(self.dim,dtype=np.float32)

    def instantiate(self,t,mean,var):
        """Instantiates normalizer based on saved statistics."""
        self.t_last = t
        self.mean = mean
        self.var = var
        if self.t_last==0:
            self.reset()
        elif self.t_last==1:
            self.std = np.abs(self.mean)
        else:
            self.std = np.sqrt(self.var)

class NormEnv:
    """Environment wrapper that normalizes observations and rewards."""

    def __init__(self,env,s_normalize,r_normalize):
        """Initialization of running normalization environment wrapper.

        Args:
            env (gym.Env): Gym environment
            s_normalize (bool): if True, normalizes observations
            r_normalize (bool): if True, normalizes rewards
        """

        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.observation_dim = gym.spaces.utils.flatdim(self.observation_space)
        self.action_dim = gym.spaces.utils.flatdim(self.action_space)

        self.s_normalize = s_normalize
        self.r_normalize = r_normalize

        self.s_rms = RunningNormalizer(self.observation_dim)
        self.r_rms = RunningNormalizer(1)
    
    def step(self,a):
        """Takes step in environment.

        Args:
            a (np.ndarray): action
        
        Returns:
            s_norm (np.ndarray): normalized next state
            r_norm (float): normalized reward
            d (bool): done flag
            info (dict): dictionary with additional environment info
        """
        s_raw, r_raw, d, info = self.env.step(a)
        
        s_norm = self.s_rms.normalize(s_raw)
        r_norm = self.r_rms.normalize(r_raw,center=False)

        self.s_raw = s_raw
        self.r_raw = r_raw

        return s_norm, r_norm, d, info
    
    def reset(self):
        """Resets environment and returns normalized initial state."""
        s_raw = self.env.reset()
        s_norm = self.s_rms.normalize(s_raw)

        self.s_raw = s_raw
        self.r_raw = None

        return s_norm

    def seed(self,seed):
        self.env.seed(seed)

    def get_raw(self):
        """Returns most recent unnormalized state and reward."""
        return self.s_raw, self.r_raw

    def update_rms(self,s_data,r_data):
        """Updates running normalization statistics."""
        if self.s_normalize and (s_data is not None):
            self.s_rms.update(s_data)
        if self.r_normalize and (r_data is not None):
            self.r_rms.update(r_data)

    def reset_rms(self):
        """Resets running normalization statistics."""
        self.s_rms.reset()
        self.r_rms.reset()

    def set_rms(self,s_t,s_mean,s_var,r_t,r_mean,r_var):
        """Sets running normalization statistics from saved values."""
        if self.s_normalize and s_t:
            assert self.s_rms.mean.shape == s_mean.shape, 's_mean shape incorrect'
            assert self.s_rms.var.shape == s_var.shape, 's_var shape incorrect'
            self.s_rms.instantiate(s_t,s_mean,s_var)
        if self.r_normalize and r_t:
            self.r_rms.instantiate(r_t,r_mean,r_var)
