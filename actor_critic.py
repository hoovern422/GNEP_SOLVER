import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np

from networks import ActorCriticNetwork
from networks import ActorNetwork
from networks import CriticNetwork

class Agent(object):    

    # The intializer
    def __init__(self, scaling_factor=1, alpha=0.0003, gamma=0.99, action_space=[-0.1,0.1], checkpoint_dir='actor'):

        self.gamma = gamma 
        self.action = None
        self.scaling_factor = scaling_factor
        self.action_space = action_space
        self.checkpoint_actor_dir = checkpoint_dir

        self.actor = ActorNetwork(checkpoint_dir=checkpoint_dir, scaling_factor=self.scaling_factor)
        self.actor.compile(optimizer=Adam(learning_rate = alpha))

####################################################################################################################         

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)

####################################################################################################################         

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)    

####################################################################################################################  

    def choose_action(self, observation, sigma):

        state = tf.convert_to_tensor([observation])
        mean = self.actor(state)

        action_distribution = tfp.distributions.Normal(loc=mean, scale=sigma)

        action = action_distribution.sample()
        action = tf.clip_by_value(action, self.action_space[0], self.action_space[1])
        self.action = action

        return action.numpy()

####################################################################################################################          

    def choose_actions(self, observations, sigma):

        states = tf.convert_to_tensor(observations)
        means = self.actor(states)
        action_distributions = tfp.distributions.Normal(loc=[means], scale=[sigma])
        actions = action_distributions.sample()
        actions = np.reshape(actions, (len(observations),))
        actions = np.clip(actions, self.action_space[0], self.action_space[1])

        return actions 

####################################################################################################################         

    def learn(self, state, reward, done, sigma, state_value, state_value_, action):

        # Convert each input into tensorflow tensors.
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            mean = self.actor(state)
            action_dist = tfp.distributions.Normal(loc=mean, scale=sigma)

            log_prob = action_dist.log_prob(action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta 

        gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            gradient, self.actor.trainable_variables))
        
####################################################################################################################           

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

####################################################################################################################         

    def save_models(self):
        self.critic.save_weights(self.critic.checkpoint_file)

####################################################################################################################         

    def load_models(self):
        self.critic.load_weights(self.critic.checkpoint_file)

####################################################################################################################         

    def get_prediction(self, state):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_value = self.critic(state)
        state_value = tf.squeeze(state_value)
        return state_value

####################################################################################################################         

    def learn(self, state, state_, reward, done):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            state_value = self.critic(state)
            state_value_ = self.critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            critic_loss = delta**2

        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient, self.critic.trainable_variables))
            
    
