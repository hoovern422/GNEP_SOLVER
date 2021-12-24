# The ML framework we'll be using.
import tensorflow as tf

# The optimizer we'll be using is Adam. Adam is an opt algo for stochastic gradient descent which is used when
# training deep learning models. Adam combines the best properties of AdaGrad and RMSProp algorithms to
# provide an optimization algorithm that can handle sparse gradients on noisy problems. Note that stochastic
# gradient descent relies on objective functions having suitable smoothness properties.
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np

from networks import ActorCriticNetwork
from networks import ActorNetwork
from networks import CriticNetwork

class Agent:

    # The intializer
    def __init__(self, scaling_factor=1, alpha=0.0003, gamma=0.99, action_space=[-0.1,0.1], checkpoint_dir='actor_critic'):

        #Save the inputs.
        self.gamma = gamma # Discount factor.
        #self.n_actions = n_actions # Number of actions an agent can take at each time-step.
        self.action = None
        self.scaling_factor = scaling_factor
        self.action_space = action_space # a list of all possible actions.
        self.checkpoint_dir = checkpoint_dir

        # Initialize the actor critic network.
        self.actor_critic = ActorCriticNetwork(checkpoint_dir=self.checkpoint_dir, scaling_factor=self.scaling_factor)

        # Compile the network with Adam as the optimizer.
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))     

    # Defines how the agent chooses an action.
    def choose_action(self, observation, sigma):

        state = tf.convert_to_tensor([observation])
        state_value, mean = self.actor_critic(state)

        action_distribution = tfp.distributions.Normal(loc=mean, scale=sigma)

        action = action_distribution.sample()
        action = tf.clip_by_value(action, self.action_space[0], self.action_space[1])

        # save the action.
        self.action = action

        return action.numpy()  

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
  
    def learn(self, state, reward, state_, done, sigma):

        #state = self.fix_shape(state)   
        #state_ = self.fix_shape(state_)

        # Convert each input into tensorflow tensors.
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed into the network.

        # Tensorflow has what's called a gradient tape, which allows you to calculate gradients manually.
        with tf.GradientTape(persistent=True) as tape:

            # Feed our state and new state through the actor critic network and get back the state values
            # for both of those states.
            #state_value, mu, sigma = self.actor_critic(state)
            state_value, mean = self.actor_critic(state)
            #state_value_, _, _ = self.actor_critic(state_)
            state_value_, _, = self.actor_critic(state_)

            # The loss function works best if it acts on a scalar value instead of a vector containing a single
            # item. So, we'll squeeze the values of both our states to get rid of the batch dimension.
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            # We need our action probabilities for the calculation of our log prob. 
            # We define the probabilities by the output of our deep neural network.
            action_dist = tfp.distributions.Normal(loc=mean, scale=sigma)

            # The log prob of the action our agent just took
            log_prob = action_dist.log_prob(self.action)

            # Calculate the TD error and the loss functions.
            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta 
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss

        # Calculate the gradients.
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))

    def choose_actions(self, observations, sigma):

        states = tf.convert_to_tensor(observations)
        _, means = self.actor_critic(states)
        action_distributions = tfp.distributions.Normal(loc=[means], scale=[sigma])
        actions = action_distributions.sample()
        actions = np.reshape(actions, (len(observations),))
        actions = np.clip(actions, self.action_space[0], self.action_space[1])

        return actions 

class NewAgent(object):    

    # The intializer
    def __init__(self, scaling_factor=1, alpha=0.0003, gamma=0.99, action_space=[-0.1,0.1], checkpoint_actor_dir='actor'):

        #Save the inputs.
        self.gamma = gamma # Discount factor.
        #self.n_actions = n_actions # Number of actions an agent can take at each time-step.
        self.action = None
        self.scaling_factor = scaling_factor
        self.action_space = action_space # a list of all possible actions.   
        self.checkpoint_actor_dir = checkpoint_actor_dir

        self.actor = ActorNetwork(checkpoint_dir=checkpoint_actor_dir, scaling_factor=self.scaling_factor)
        self.actor.compile(optimizer=Adam(learning_rate = alpha))

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)

    def choose_action(self, observation, sigma):

        state = tf.convert_to_tensor([observation])
        mean = self.actor(state)

        action_distribution = tfp.distributions.Normal(loc=mean, scale=sigma)

        action = action_distribution.sample()
        action = tf.clip_by_value(action, self.action_space[0], self.action_space[1])

        # save the action.
        self.action = action

        return action.numpy()     

    def choose_actions(self, observations, sigma):

        states = tf.convert_to_tensor(observations)
        means = self.actor(states)
        action_distributions = tfp.distributions.Normal(loc=[means], scale=[sigma])
        actions = action_distributions.sample()
        actions = np.reshape(actions, (len(observations),))
        actions = np.clip(actions, self.action_space[0], self.action_space[1])

        return actions    

    def learn(self, state, reward, done, sigma, state_value, state_value_, action):

        # Convert each input into tensorflow tensors.
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            mean = self.actor(state)
            action_dist = tfp.distributions.Normal(loc=mean, scale=sigma)

            log_prob = action_dist.log_prob(action)

            # Calculate the TD error and the loss functions.
            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta 

        # Calculate the gradients.
        gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            gradient, self.actor.trainable_variables))   

class Critic(object):

    # The intializer
    def __init__(self, alpha=0.0003, gamma=0.99, checkpoint_critic_dir='critic'):

        #Save the inputs.
        self.gamma = gamma # Discount factor.
        #self.n_actions = n_actions # Number of actions an agent can take at each time-step.
        self.action = None
        self.checkpoint_actor_dir = checkpoint_critic_dir

        self.critic = CriticNetwork(checkpoint_dir=checkpoint_critic_dir)
        self.critic.compile(optimizer=Adam(learning_rate = alpha))

    def save_models(self):
        print('... saving models ...')
        self.critic.save_weights(self.actor.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.critic.load_weights(self.actor.checkpoint_file)

    def get_prediction(self, state):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_value = self.critic(state)
        state_value = tf.squeeze(state_value)
        return state_value

    def learn(self, state, state_, reward, done):

        # Convert each input into tensorflow tensors.
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            state_value = self.critic(state)
            state_value_ = self.critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            # Calculate the TD error and the loss functions.
            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            critic_loss = delta**2

        # Calculate the gradients.
        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient, self.critic.trainable_variables))
            
    