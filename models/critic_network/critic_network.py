import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense

##############################################################################


class CriticNetwork(keras.Model):

    """
    The critic network used in the actor-critic GNEP solver.
    """

    def __init__(
            self,
            fc1_dims=1024,
            fc2_dims=512,
            name='critic',
            checkpoint_dir='tmp/actor_critic'):

        super(CriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, name + '_critic')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.state_value = Dense(1, activation=None)

##############################################################################

    def call(self, state):
        "Passes a state through the critic network."

        val = self.fc1(state)
        val = self.fc2(val)
        state_value = self.state_value(val)

        return state_value
