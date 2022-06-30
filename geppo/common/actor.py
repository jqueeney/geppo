import gym
import numpy as np
import tensorflow as tf

from geppo.common.ac_utils import transform_features, create_nn 
from geppo.common.ac_utils import flat_to_list, list_to_flat

class Actor:
    """Base policy class."""

    def __init__(self,env):
        """Initializes policy.

        Args:
            env (NormEnv): normalized environment
        """

        self.s_dim = env.observation_dim
        self.a_dim = env.action_dim

    def _forward(self,s):
        """Returns output of neural network."""
        raise NotImplementedError

    def _forward_pik(self,s):
        """Returns output of pi_k neural network."""
        raise NotImplementedError

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state."""
        raise NotImplementedError

    def clip(self,a):
        """Clips action to feasible range."""
        raise NotImplementedError
    
    def neglogp(self,s,a):
        """Calculates negative log probability for given state and action."""
        raise NotImplementedError

    def neglogp_pik(self,s,a):
        """Same as neglogp but for pi_k policy."""
        raise NotImplementedError

    def entropy(self,s):
        """Calculates entropy of current policy."""
        raise NotImplementedError

    def kl(self,s,kl_info_ref,direction='forward'):
        """Calculates KL divergence between current and reference policy."""
        raise NotImplementedError

    def get_kl_info(self,s):
        """Returns info needed to calculate KL divergence."""
        raise NotImplementedError

    def get_weights(self,flat=False):
        """Returns parameter weights of current policy.
        
        Args:
            flat (bool): if True, returns weights as flattened np.ndarray
        
        Returns:
            list or np.ndarray of parameter weights.
        """
        raise NotImplementedError
    
    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights of current policy.
        
        Args:
            weights (list, np.ndarray): list or np.ndarray of parameter weights
            from_flat (bool): if True, weights are flattened np.ndarray
            increment (bool): if True, weights are incremental values
        """
        raise NotImplementedError
    
    def update_pik_weights(self):
        """Copies weights of current policy to pi_k policy."""
        raise NotImplementedError

class GaussianActor(Actor):
    """Multivariate Gaussian policy with diagonal covariance.

    Mean action for a given state is parameterized by a neural network.
    Diagonal covariance parameterized by same neural network if state-dependent,
    otherwise is parameterized separately.
    """

    def __init__(self,env,layers,activations,gain,std_mult=1.0,
        per_state_std=False):
        """Initializes multivariate Gaussian policy with diagonal covariance.

        Args:
            env (NormEnv): normalized environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer initialization
            std_mult (float): multiplicative factor for diagonal covariance 
                initialization
            per_state_std (bool): if True, state-dependent diagonal covariance
        """

        assert isinstance(env.action_space,gym.spaces.Box), (
            'Only Box action space supported')
        
        super(GaussianActor,self).__init__(env)

        self.per_state_std = per_state_std
        
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        self.logstd_init = np.ones((1,)+env.action_space.shape,
            dtype='float32') * np.log(std_mult)

        if self.per_state_std:
            self._nn = create_nn(self.s_dim,2*self.a_dim,layers,activations,
                gain,name='actor')
            self.trainable = self._nn.trainable_variables

            # Stores most recent starting policy pi_k
            self._nn_pik = create_nn(self.s_dim,2*self.a_dim,layers,activations,
                gain,name='actor_pik')
        else:
            self._nn = create_nn(self.s_dim,self.a_dim,layers,activations,gain,
                name='actor_mean')
            self.logstd = tf.Variable(np.zeros_like(self.logstd_init),
                dtype=tf.float32,name='logstd')
            self.trainable = self._nn.trainable_variables + [self.logstd]

            # Stores most recent starting policy pi_k
            self._nn_pik = create_nn(self.s_dim,self.a_dim,layers,activations,
                gain,name='actor_mean_pik')
            self.logstd_pik = tf.Variable(np.zeros_like(self.logstd_init),
                dtype=tf.float32,name='logstd_pik')

        self.update_pik_weights()

    def _forward(self,s):
        """Returns mean actions and log std from neural network."""
        s_feat = transform_features(s)
        a_out = self._nn(s_feat)
        if self.per_state_std:
            a_mean, a_logstd = tf.split(a_out,num_or_size_splits=2,axis=-1)
        else:
            a_mean = a_out
            a_logstd = self.logstd * tf.ones_like(a_mean)
        
        a_logstd = a_logstd + self.logstd_init
        a_logstd = tf.maximum(a_logstd,tf.math.log(1e-3))

        return a_mean, a_logstd

    def _forward_pik(self,s):
        """Returns mean actions and log std from pi_k neural network."""
        s_feat = transform_features(s)
        a_out = self._nn_pik(s_feat)
        if self.per_state_std:
            a_mean, a_logstd = tf.split(a_out,num_or_size_splits=2,axis=-1)
        else:
            a_mean = a_out
            a_logstd = self.logstd_pik * tf.ones_like(a_mean)
        
        a_logstd = a_logstd + self.logstd_init
        a_logstd = tf.maximum(a_logstd,tf.math.log(1e-3))

        return a_mean, a_logstd

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state.
        
        Args:
            s (np.ndarray): state
            deterministic (bool): if True, returns mean action
        
        Returns:
            Action sampled from current policy.
        """
        act, a_logstd = self._forward(s)

        if not deterministic:
            u = np.random.normal(size=np.shape(act))
            act = act + tf.exp(a_logstd) * u

        if np.shape(act)[0] == 1:
            act = tf.squeeze(act,axis=0)

        return act

    def clip(self,a):
        return np.clip(a,self.act_low,self.act_high)
    
    def neglogp(self,s,a):
        a_mean, a_logstd = self._forward(s)

        a_vec = (tf.square((a - a_mean) / tf.exp(a_logstd)) 
            + 2*a_logstd + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    def neglogp_pik(self,s,a):
        a_mean, a_logstd = self._forward_pik(s)

        a_vec = (tf.square((a - a_mean) / tf.exp(a_logstd)) 
            + 2*a_logstd + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    def entropy(self,s):
        _, a_logstd = self._forward(s)
        vec = 2*a_logstd + tf.math.log(2*np.pi) + 1
        return 0.5 * tf.reduce_sum(vec,axis=-1)

    def kl(self,s,kl_info_ref,direction='forward'):
        """Calculates KL divergence between current and reference policy.
        
        Args:
            s (np.ndarray): states
            kl_info_ref (tuple): mean actions and log std. deviation for 
                reference policy
            direction (string): forward or reverse
        
        Returns:
            np.ndarray of KL divergences between current policy and reference 
            policy at every input state.
        """
        mean_ref, logstd_ref = np.moveaxis(kl_info_ref,-1,0)
        a_mean, a_logstd = self._forward(s)

        if direction == 'forward':
            num = tf.square(a_mean-mean_ref) + tf.exp(2*logstd_ref)
            vec = num / tf.exp(2*a_logstd) + 2*a_logstd - 2*logstd_ref - 1
        else:
            num = tf.square(a_mean-mean_ref) + tf.exp(2*a_logstd)
            vec = num / tf.exp(2*logstd_ref) + 2*logstd_ref - 2*a_logstd - 1

        return 0.5 * tf.reduce_sum(vec,axis=-1)

    def get_kl_info(self,s):
        mean_ref, logstd_ref = self._forward(s)
        return np.stack((mean_ref,logstd_ref),axis=-1)

    def get_weights(self,flat=False):
        weights = self._nn.get_weights()
        if not self.per_state_std:
            weights = weights + [self.logstd.numpy()]
        
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False):
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))
        
        if self.per_state_std:
            self._nn.set_weights(weights)
        else:
            model_weights = weights[:-1]
            logstd_weights = weights[-1]
            logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
            
            self._nn.set_weights(model_weights)
            self.logstd.assign(logstd_weights)
    
    def update_pik_weights(self):
        model_weights = self._nn.get_weights() 
        self._nn_pik.set_weights(model_weights)
        
        if not self.per_state_std:
            logstd_weights = self.logstd.numpy()
            self.logstd_pik.assign(logstd_weights)

class SoftMaxActor(Actor):
    """Softmax policy for discrete action spaces."""

    def __init__(self,env,layers,activations,gain):
        """Initializes softmax policy.
        
        Args:
            env (NormEnv): normalized environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer initialization
        """

        assert isinstance(env.action_space,gym.spaces.Discrete), (
            'Only Discrete action space supported')
        
        super(SoftMaxActor,self).__init__(env)

        self._nn = create_nn(self.s_dim,self.a_dim,layers,activations,gain,
            name='actor')

        self.trainable = self._nn.trainable_variables

        # Stores most recent starting policy pi_k
        self._nn_pik = create_nn(self.s_dim,self.a_dim,layers,activations,gain,
            name='actor_pik')        
        self.update_pik_weights()

    def _forward(self,s):
        """Returns logits from neural network."""
        s_feat = transform_features(s)
        return self._nn(s_feat)

    def _forward_pik(self,s):
        """Returns logits from pi_k neural network."""
        s_feat = transform_features(s)
        return self._nn_pik(s_feat)

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state.
        
        Args:
            s (np.ndarray): state
            deterministic (bool): if True, returns arg max
        
        Returns:
            Action sampled from current policy.
        """
        a_logits = self._forward(s)
        a_logits = a_logits - tf.reduce_max(a_logits,axis=-1,keepdims=True)

        if deterministic:
            act = tf.argmax(a_logits,axis=-1)
        else:
            # Gumbel trick
            u = np.random.random(size=np.shape(a_logits))
            act = tf.argmax(a_logits - np.log(-np.log(u)),axis=-1)
        
        return tf.squeeze(act)

    def clip(self,a):
        return a
    
    def neglogp(self,s,a):
        a_logits = self._forward(s)
        a_logits = a_logits - tf.reduce_max(a_logits,axis=-1,keepdims=True)

        a_labels = tf.one_hot(a,a_logits.shape[-1])

        neglogp = tf.nn.softmax_cross_entropy_with_logits(a_labels,a_logits)
        return tf.squeeze(neglogp)

    def neglogp_pik(self,s,a):
        a_logits = self._forward_pik(s)
        a_logits = a_logits - tf.reduce_max(a_logits,axis=-1,keepdims=True)

        a_labels = tf.one_hot(a,a_logits.shape[-1])

        neglogp = tf.nn.softmax_cross_entropy_with_logits(a_labels,a_logits)
        return tf.squeeze(neglogp)

    def entropy(self,s):
        logits_cur = self._forward(s)
        logits_cur = logits_cur - tf.reduce_max(
            logits_cur,axis=-1,keepdims=True)
        logsumexp_cur = tf.reduce_logsumexp(logits_cur,axis=-1,keepdims=True)

        prob_num = tf.exp(logits_cur)
        prob_den = tf.reduce_sum(prob_num,axis=-1,keepdims=True)
        prob = prob_num / prob_den

        return tf.reduce_sum(prob * (logits_cur - logsumexp_cur),axis=-1) * -1

    def kl(self,s,kl_info_ref,direction='forward'):
        """Calculates KL divergence between current and reference policy.
        
        Args:
            s (np.ndarray): states
            kl_info_ref (tuple): logits for reference policy
            direction (string): forward or reverse
        
        Returns:
            np.ndarray of KL divergences between current policy and reference 
            policy at every input state.
        """
        logits_ref = kl_info_ref
        
        logits_cur = self._forward(s)
        logits_cur = logits_cur - tf.reduce_max(
            logits_cur,axis=-1,keepdims=True)
        logsumexp_cur = tf.reduce_logsumexp(logits_cur,axis=-1,keepdims=True)

        logits_ref = logits_ref - tf.reduce_max(
            logits_ref,axis=-1,keepdims=True)
        logsumexp_ref = tf.reduce_logsumexp(logits_ref,axis=-1,keepdims=True)

        val = (logits_cur - logsumexp_cur) - (logits_ref - logsumexp_ref)
        
        if direction == 'forward':
            prob_num = tf.exp(logits_cur)
        else:
            prob_num = tf.exp(logits_ref)
            val = val * -1

        prob_den = tf.reduce_sum(prob_num,axis=-1,keepdims=True)
        prob = prob_num / prob_den

        return tf.reduce_sum(prob * val,axis=-1)

    def get_kl_info(self,s):
        logits_ref = self._forward(s).numpy()
        return logits_ref

    def get_weights(self,flat=False):
        weights = self._nn.get_weights()
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False):
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))
        
        self._nn.set_weights(weights)
    
    def update_pik_weights(self):
        weights = self._nn.get_weights() 
        self._nn_pik.set_weights(weights)