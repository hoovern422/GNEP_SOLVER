import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense

##############################################################################

class ActorNetwork(keras.Model):

    "The actor network used by the agent in the actor-critic GNEP solver."

    def __init__(
            self,
            action_scaling_factor=1,
            fc1_dims=1024,
            fc2_dims=512,
            name='critic',
            checkpoint_dir='tmp/actor_critic'):

        super(ActorNetwork, self).__init__()

        self.action_scaling_factor = action_scaling_factor
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, name + '_actor')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mean = Dense(1, activation='tanh')

##########################################################################

    def call(self, state):

        "Passes a state through the actor network."

        val = self.fc1(state)
        val = self.fc2(val)
        mean = self.mean(val) * self.action_scaling_factor

        return mean

##############################################################################
