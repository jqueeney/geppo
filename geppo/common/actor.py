import gym
import numpy as np
import tensorflow as tf

from geppo.common.ac_utils import transform_features, create_nn 
from geppo.common.ac_utils import flat_to_list, list_to_flat

class GaussianActor:
    """Multivariate Gaussian policy with diagonal covariance.

    Mean action for a given state is parameterized by a neural network.
    Diagonal covariance is state-independent and parameterized separately.
    """

    def __init__(self,env,layers,activations,gain,std_mult=1.0):
        """Initializes multivariate Gaussian policy with diagonal covariance.

        Args:
            env (NormEnv): normalized Gym environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer initialization
            std_mult (float): multiplicative factor for diagonal covariance 
                initialization as multiple of half of action range
        """

        assert isinstance(env.action_space,gym.spaces.Box), (
            'Only Box action space supported')
        
        in_dim = env.observation_space.shape[0]
        out_dim = env.action_space.shape[0]
        self._nn = create_nn(in_dim,out_dim,layers,activations,gain,
            name='actor')

        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        if any(np.isinf(self.act_low)) or any(np.isinf(self.act_high)):
            logstd_init = np.zeros((1,)+env.action_space.shape)
        else:
            logstd_mult = np.log(std_mult*((self.act_high - self.act_low) / 2))
            logstd_init = np.ones((1,)+env.action_space.shape) * logstd_mult

        self.logstd = tf.Variable(logstd_init,dtype=tf.float32,name='logstd')

        self.trainable = self._nn.trainable_variables + [self.logstd]

        # Stores most recent starting policy pi_k
        self._nn_pik = create_nn(in_dim,out_dim,layers,activations,gain,
            name='actor_pik')
        self.logstd_pik = tf.Variable(logstd_init,dtype=tf.float32,
            name='logstd_pik')
        self.update_pik_weights()

    def _forward(self,s):
        """Returns output of neural network."""
        s_feat = transform_features(s)
        return self._nn(s_feat)

    def _forward_pik(self,s):
        """Returns output of pi_k neural network."""
        s_feat = transform_features(s)
        return self._nn_pik(s_feat)

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state.
        
        Args:
            s (np.ndarray): state
            deterministic (bool): if True, returns mean action
        
        Returns:
            Action sampled from current policy.
        """
        a_mean = self._forward(s)

        if deterministic:
            act = a_mean
        else:
            u = tf.random.normal(tf.shape(a_mean), dtype=a_mean.dtype)
            act = a_mean + tf.exp(self.logstd) * u

        if act.shape[0] == 1:
            act = tf.squeeze(act,axis=0)

        return act

    def clip(self,a):
        """Clips action to feasible range."""
        return tf.clip_by_value(a,self.act_low,self.act_high)
    
    def neglogp(self,s,a):
        """Calculates negative log probability for given state and action."""
        a_mean = self._forward(s)

        a_vec = (tf.square((a - a_mean) / tf.exp(self.logstd)) 
            + 2*self.logstd + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    def neglogp_pik(self,s,a):
        """Same as neglogp but for pi_k policy."""
        a_mean = self._forward_pik(s)

        a_vec = (tf.square((a - a_mean) / tf.exp(self.logstd_pik)) 
            + 2*self.logstd_pik + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    def entropy(self,s):
        """Calculates entropy of current policy."""
        vec = 2*self.logstd + tf.math.log(2*np.pi) + 1
        ent = 0.5 * tf.reduce_sum(vec)
        return ent * tf.ones((s.shape[0]),dtype=tf.float32)

    def kl(self,s,kl_info_ref):
        """Calculates KL divergence between current policy and reference policy.
        
        Args:
            s (np.ndarray): states
            kl_info_ref (tuple): mean actions and log std. deviation for 
                reference policy.
        
        Returns:
            np.ndarray of KL divergences between current policy and reference 
            policy at every input state.
        """
        mean_ref, logstd_ref = kl_info_ref
        a_mean = self._forward(s)

        num = tf.square(a_mean-mean_ref) + tf.exp(2*logstd_ref)
        vec = num / tf.exp(2*self.logstd) + 2*self.logstd - 2*logstd_ref - 1

        return 0.5 * tf.reduce_sum(vec,axis=-1)

    def get_kl_info(self,s):
        """Returns info needed to calculate KL divergence."""
        mean_ref = self._forward(s).numpy()
        logstd_ref = self.logstd.numpy()
        return mean_ref, logstd_ref

    def get_weights(self,flat=False):
        """Returns parameter weights of current policy.
        
        Args:
            flat (bool): if True, returns weights as flattened np.ndarray
        
        Returns:
            list or np.ndarray of parameter weights.
        """
        weights = self._nn.get_weights() + [self.logstd.numpy()]
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights of current policy.
        
        Args:
            weights (list, np.ndarray): list or np.ndarray of parameter weights
            from_flat (bool): if True, weights are flattened np.ndarray
            increment (bool): if True, weights are incremental values
        """
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))
        
        model_weights = weights[:-1]
        logstd_weights = weights[-1]
        logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
        
        self._nn.set_weights(model_weights)
        self.logstd.assign(logstd_weights)
    
    def update_pik_weights(self):
        """Copies weights of current policy to pi_k policy."""
        model_weights = self._nn.get_weights() 
        logstd_weights = self.logstd.numpy()

        self._nn_pik.set_weights(model_weights)
        self.logstd_pik.assign(logstd_weights)