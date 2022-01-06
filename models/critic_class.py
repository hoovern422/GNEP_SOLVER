import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

##############################################################################

class CriticNetwork(keras.Model):

    """
    The critic network used by the critic in the actor-critic GNEP solver.
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

##############################################################################


class Critic(object):

    """
    The critic class contain a critic network that judges the actions chosen
    by the actor networks.
    """

    def __init__(self, alpha=0.0003, gamma=0.99,
                 checkpoint_critic_dir='critic'):

        self.gamma = gamma  # Discount factor.
        self.action = None
        self.checkpoint_actor_dir = checkpoint_critic_dir

        self.critic = CriticNetwork(checkpoint_dir=checkpoint_critic_dir)
        self.critic.compile(optimizer=Adam(learning_rate=alpha))

##############################################################################

    def save_models(self):
        """
        Saves critic network paramaters.
        """

        self.critic.save_weights(self.critic.checkpoint_file)

##############################################################################

    def load_models(self):
        """
        Saves critic network paramaters.
        """

        self.critic.load_weights(self.critic.checkpoint_file)

##############################################################################

    def get_prediction(self, state):
        """
        Get the critic network's prediction based on the current state.
        """

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_value = self.critic(state)
        state_value = tf.squeeze(state_value)
        return state_value

##############################################################################

    def learn(self, state, state_, reward, done):
        """
        Updates critic network paramaters using the ADAM opt. algorithm.
        """

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            state_value = self.critic(state)
            state_value_ = self.critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            delta = reward + self.gamma * state_value_ * \
                (1 - int(done)) - state_value
            critic_loss = delta**2

        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient, self.critic.trainable_variables))
