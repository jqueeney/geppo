import tensorflow as tf

from geppo.common.ac_utils import transform_features
from geppo.common.ac_utils import create_nn

class Critic:
    """Value function used for advantage function estimation."""

    def __init__(self,env,layers,activations,gain):
        """Initializes value function.

        Args:
            env (NormEnv): normalized environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer 
                initialization
        """        

        in_dim = env.observation_dim
        self._nn = create_nn(in_dim,1,layers,activations,gain,name='critic')

        self.trainable = self._nn.trainable_variables

    def _forward(self,s):
        """Returns output of neural network."""
        s_feat = transform_features(s)
        return self._nn(s_feat)

    def value(self,s):
        """Calculates value given the state."""
        return tf.squeeze(self._forward(s),axis=-1)

    def get_weights(self):
        """Returns parameter weights."""
        return self._nn.get_weights()

    def set_weights(self,weights):
        """Sets parameter weights."""
        self._nn.set_weights(weights)