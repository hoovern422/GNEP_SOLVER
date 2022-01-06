import numpy as np
import timeit
from models.critic_class import Critic
from models.agent_class import Agent
from GNEP_environments.env2.training_environment import TrainingEnv
from GNEP_environments.env2.playing_environment import PlayingEnv
from GNEP_algorithms.training_algorithm import train
from GNEP_algorithms.playing_algorithm import play
from plots.plotters import plot_heatmap, plot_learning_curve, plot_nash_eq

##############################################################################

if __name__ == '__main__':

    """
    the main function executes the training and playing processes of the
    algorithm employed in the GNEP solver.
    """
    tic = timeit.default_timer()

    training_flag = True
    playing_flag = True

    # Define GNEP-specific function parameters.
    x_bounds = np.array([0, 1])
    y_bounds = np.array([0, 1])
    action_space = [-0.1, 0.1]
    action_scaling_factor = 0.1
    reward_scaling_factor = 1
    training_threshold = 0.01
    playing_threshold = 0.005
    max_train_games = 200

    # Define neural network hyperparamaters
    actor_lr = 1e-5
    critic_lr = 1e-5 
    gamma = 0.99

    # Instantiate an agent for each player in the GNEP (each with an actor
    # network).
    x_agent = Agent(action_scaling_factor, actor_lr, gamma, action_space,
                    checkpoint_dir='models/problem2_checkpoints/x_agent')
    y_agent = Agent(action_scaling_factor, actor_lr, gamma, action_space,
                    checkpoint_dir='models/problem2_checkpoints/y_agent')

    if training_flag:
        
        # Instantiate a critic and train the actor networks.
        env = TrainingEnv(x_bounds, y_bounds, training_threshold,
                          reward_scaling_factor)
        critic = Critic(critic_lr, gamma,
                        checkpoint_critic_dir='models/problem2_checkpoints/critic')

        x_sols, y_sols, avg_rewards = train(
            env, x_agent, y_agent, critic, max_train_games)

        episode_tracker = [i + 1 for i in range(len(avg_rewards))]
        plot_learning_curve(episode_tracker, avg_rewards, 'problem2')

    if playing_flag:

        # Define how many agents there will be and how many games they will 
        # play in parallel.
        num_agents = 25000
        num_play_games = 2

        env = PlayingEnv(x_bounds, y_bounds, num_agents,
                        num_play_games, playing_threshold)

        x_sols, y_sols = play(env, x_agent, y_agent, num_agents, num_play_games)

        plot_nash_eq(x_sols, y_sols, 'problem2')
        plot_heatmap(x_sols, y_sols, 'problem2')

        toc = timeit.default_timer()
        time = toc - tic

        print("")
        print("Number of solutions found with distance <=",
            playing_threshold, ": ", len(x_sols))
        print("Total time in seconds:", time)

##############################################################################