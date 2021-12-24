import numpy as np
import actor_critic as ac_API
from actor_critic import Agent, NewAgent, Critic
from training_env import GNEP_ENV
from playing_env import GNEP_ENV_MULTI
import matplotlib.pyplot as plt
import math
from jax import numpy as jnp
from jax import random as jrandom
import timeit

if __name__ == '__main__':

#########################################################################################################
    def plot_learning_curve(x, scores):

      running_avg = np.zeros(len(scores))
      for i in range(len(running_avg)):
          running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
      plt.plot(x, running_avg)
      plt.title('Running average of previous 100 scores')
      plt.show()           
#############################################################################################################

    tic = timeit.default_timer()

    x_bounds = np.array([0,10])
    y_bounds = np.array([0,10])
    max_games = 200
    action_scaling_factor = 0.5
    reward_scaling_factor = 1/15
    action_space = [-0.5, 0.5]
    alpha = 1e-5
    training_threshold = 0.1
    playing_threshold = 0.01
    
    env = TRAINING_ENV(x_bounds, y_bounds, training_threshold, reward_scaling_factor)
    x_agent = Agent(scaling_factor=action_scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_dir='tmp/actor_x_agent')
    y_agent = Agent(scaling_factor=action_scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_dir='tmp/actor_y_agent')
    critic = Critic(alpha=alpha, gamma=0.999, checkpoint_critic_dir='tmp/critic')

    x_sols1, y_sols1 = train(env, x_agent, y_agent, critic, max_games)

    num_agents = 50000
    num_games = 2

    env = PLAYING_ENV(x_bounds, y_bounds, num_agents=num_agents, num_games=num_games, threshold=playing_threshold)
    x_sols2, y_sols2 = play(env, x_agent, y_agent, num_agents)

    x_sols2 = np.array(x_sols2)
    y_sols2 = np.array(y_sols2)

    toc = timeit.default_timer()
    time = toc - tic
    print("")
    print("Number of solutions found with distance <= 0.01:", len(x_sols2))
    print("Total time in seconds:", time)  

    plt.title('Plot for Nash Equilibria (Playing)')
    plt.scatter(x_sols2, y_sols2, c='k', marker='.')
    plt.show()

    data = np.vstack([x_sols2, y_sols2])
    z = gaussian_kde(data)(data)
    idx = z.argsort()
    x_sols2, y_sols2, z = x_sols2[idx], y_sols2[idx], z[idx]
    fig, ax = plt.subplots()
    cax = ax.scatter(x_sols2, y_sols2, c=z, s=100)
    plt.colorbar(cax)
    plt.title("Heatmap ")
    plt.show()
    
#############################################################################################################if __name__ == '__main__':

#########################################################################################################
    def plot_learning_curve(x, scores):

      running_avg = np.zeros(len(scores))
      for i in range(len(running_avg)):
          running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
      plt.plot(x, running_avg)
      plt.title('Running average of previous 100 scores')
      plt.show()           
#############################################################################################################

    tic = timeit.default_timer()

    x_bounds = np.array([0,10])
    y_bounds = np.array([0,10])
    max_games = 200
    action_scaling_factor = 0.5
    reward_scaling_factor = 1/15
    action_space = [-0.5, 0.5]
    alpha = 1e-5
    training_threshold = 0.1
    playing_threshold = 0.01
    
    env = TRAINING_ENV(x_bounds, y_bounds, training_threshold, reward_scaling_factor)
    x_agent = Agent(scaling_factor=action_scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_dir='tmp/actor_x_agent')
    y_agent = Agent(scaling_factor=action_scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_dir='tmp/actor_y_agent')
    critic = Critic(alpha=alpha, gamma=0.999, checkpoint_critic_dir='tmp/critic')

    x_sols1, y_sols1 = train(env, x_agent, y_agent, critic, max_games)

    num_agents = 50000
    num_games = 2

    env = PLAYING_ENV(x_bounds, y_bounds, num_agents=num_agents, num_games=num_games, threshold=playing_threshold)
    x_sols2, y_sols2 = play(env, x_agent, y_agent, num_agents)

    x_sols2 = np.array(x_sols2)
    y_sols2 = np.array(y_sols2)

    toc = timeit.default_timer()
    time = toc - tic
    print("")
    print("Number of solutions found with distance <= 0.01:", len(x_sols2))
    print("Total time in seconds:", time)  

    plt.title('Plot for Nash Equilibria (Playing)')
    plt.scatter(x_sols2, y_sols2, c='k', marker='.')
    plt.show()

    data = np.vstack([x_sols2, y_sols2])
    z = gaussian_kde(data)(data)
    idx = z.argsort()
    x_sols2, y_sols2, z = x_sols2[idx], y_sols2[idx], z[idx]
    fig, ax = plt.subplots()
    cax = ax.scatter(x_sols2, y_sols2, c=z, s=100)
    plt.colorbar(cax)
    plt.title("Heatmap ")
    plt.show()
    
#############################################################################################################
