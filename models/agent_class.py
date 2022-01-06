import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from models.neural_networks.actor_network import ActorNetwork

class Agent(object):

    """
    The agent class contains an actor network that determines the actions of
    the variables that agent controls.
    """

    def __init__(self, action_scaling_factor=1, alpha=0.0003, gamma=0.99,
                 action_space=[-0.1, 0.1], checkpoint_dir='actor'):

        self.gamma = gamma
        self.action = None
        self.action_scaling_factor = action_scaling_factor
        self.action_space = action_space
        self.checkpoint_dir = checkpoint_dir

        self.actor = ActorNetwork(
            checkpoint_dir=self.checkpoint_dir,
            action_scaling_factor=self.action_scaling_factor)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))

##########################################################################

    def save_models(self):
        """
        Saves actor network paramaters.
        """

        self.actor.save_weights(self.actor.checkpoint_file)

##########################################################################

    def load_models(self):
        """
        Loads actor network paramaters.
        """

        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)

##########################################################################

    def choose_action(self, state, sigma):
        """
        Chooses an action based the current state.
        """

        state = tf.convert_to_tensor([state])
        mean = self.actor(state)

        action_distribution = tfp.distributions.Normal(loc=mean, scale=sigma)

        action = action_distribution.sample()
        action = tf.clip_by_value(
            action,
            self.action_space[0],
            self.action_space[1])
        self.action = action

        return action.numpy()

##########################################################################

    def choose_actions(self, states, sigma):
        """
        Chooses a set of actions based on current states.
        """

        states = tf.convert_to_tensor(states)
        means = self.actor(states)
        action_distributions = tfp.distributions.Normal(
            loc=[means], scale=[sigma])
        actions = action_distributions.sample()
        actions = np.reshape(actions, (len(states),))
        actions = np.clip(actions, self.action_space[0], self.action_space[1])

        return actions

##########################################################################

    def learn(
            self,
            state,
            reward,
            done,
            sigma,
            state_value,
            state_value_,
            action):
        """
        Updates actor network paramaters using the ADAM opt. algorithm.
        """

        # Convert each input into tensorflow tensors.
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            mean = self.actor(state)
            action_dist = tfp.distributions.Normal(loc=mean, scale=sigma)

            log_prob = action_dist.log_prob(action)

            delta = reward + self.gamma * state_value_ * \
                (1 - int(done)) - state_value
            actor_loss = -log_prob * delta

        gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            gradient, self.actor.trainable_variables))
