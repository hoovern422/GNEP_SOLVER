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

    def plot_learning_curve(x, scores, best_observations_x, best_observations_y):

      plt.title('Plot for Nash Equilibria')
      plt.scatter(best_observations_x, best_observations_y, c='k', marker='.')
      plt.show()

      running_avg = np.zeros(len(scores))
      for i in range(len(running_avg)):
          running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
      plt.plot(x, running_avg)
      plt.title('Running average of previous 100 scores')
      plt.show()

#########################################################################################################

    def obj_func1(x,y):
        
        return x**2 + (8/3)*x*y - 34*x

#########################################################################################################

    def obj_func2(x,y):
        
        return y**2 + (5/4)*x*y - 24.25*y   

#########################################################################################################        
    
    def constraint_func(other):
        
        return (15 - other)

###########################################################################################

    def train(env, x_agent, y_agent, critic, max_games):

        game_counter = 0
        games_since_last_save = 0

        done = False

        best_score = -100
        avg_score = -100

        score_history = []

        best_observations_x = []
        best_observations_y = []   

        while not done:

            game_counter += 1
            games_since_last_save += 1  

            step_counter = 0
            
            observation = env.reset()

            game_over = False
            score = 0
            is_nash_eq = False
            x_move = True
            y_move = False
            sigma = 1 / game_counter

            while not game_over:

                step_counter += 1

                if (step_counter % 2 == 0):

                    agent = x_agent
                    x_move = True
                    y_mve = False

                else:

                    agent = y_agent
                    x_move = True
                    y_move = True

                action = agent.choose_action(observation, sigma)[0][0]#, std_dev)   

                new_x, new_y, reward, game_over, is_nash_eq = env.step(action, x_move, y_move)
                observation_ = [new_x, new_y]
                score += reward     

                critic.learn(observation, observation_, reward, done)
                state_value = critic.get_prediction(observation)
                state_value_ = critic.get_prediction(observation_)
                agent.learn(observation, reward, done, sigma, state_value, state_value_, action)

                # Update the observation
                observation = observation_

            # At the end of the game update the score
            score = score / step_counter
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])    

            if (is_nash_eq and game_counter >= 50):

                agent.save_models()
                critic.save_models()
                games_since_last_save = 0

            if is_nash_eq:

                best_observations_x.append(observation[0])
                best_observations_y.append(observation[1])
                print(observation[0], observation[1])
                is_nash_eq = False      

            if (game_counter == max_games or games_since_last_save >= 100):

                done = True 

            print('episode ', game_counter, 'score %.3f' % score, 'avg_score %.3f' % avg_score)    

        x = [i+1 for i in range(game_counter)]                                                      
        plot_learning_curve(x, score_history, best_observations_x, best_observations_y)

        return best_observations_x, best_observations_y, sigma           

#########################################################################################################

    def play(env, x_agent, y_agent, num_players):

        done = False
        best_observations_x = []
        best_observations_y = []
        x_move = True
        y_move = False

        nash_eq_vec = np.zeros(num_players)
        game_counter_vec = np.zeros(num_players)

        x_agent.load_models()    
        y_agent.load_models()   
        env.game_over_vec = np.ones(num_players)

        x_observations, y_observations = env.reset()

        sigma = 1e-3

        step_counter = 0

        while not done:

            step_counter += 1

            observations = np.stack((x_observations, y_observations), axis=-1)

            if (step_counter %2 == 0):

                agent = x_agent
                x_move = True
                y_move = False

            else:
                agent = y_agent
                y_move = True
                x_move = False    

            actions = agent.choose_actions(observations, sigma)       

            x_observations, y_observations, nash_eq_vec, game_counter_vec = env.step(actions, x_move, y_move)

            nash_eq_indices = [i for i,j in enumerate(nash_eq_vec) if j == 1]
            best_observations_x.extend(x_observations[nash_eq_indices])
            best_observations_y.extend(y_observations[nash_eq_indices])
     
            done = (np.sum(game_counter_vec) >= num_players * num_games)  

            if not done:
                x_observations, y_observations = env.reset()                

        return best_observations_x, best_observations_y 
#############################################################################################################

    x_bounds = np.array([0,10])
    y_bounds = np.array([0,10])
    scaling_factor = 0.5
    action_space = [-0.5, 0.5]
    alpha = 1e-5

    tic = timeit.default_timer()
    
    env = GNEP_ENV(obj_func1, obj_func2, x_bounds, y_bounds, constraint_func)
    
    #x_agent = Agent(scaling_factor=scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_dir='tmp/actor_critic_x_agent')
    #y_agent = Agent(scaling_factor=scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_dir='tmp/actor_critic_y_agent')
    x_agent = NewAgent(scaling_factor=scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_actor_dir='tmp/actor_x_agent')
    y_agent = NewAgent(scaling_factor=scaling_factor, alpha=alpha, gamma=0.999, action_space=action_space, checkpoint_actor_dir='tmp/actor_y_agent')
    critic = Critic(alpha=alpha, gamma=0.999, checkpoint_critic_dir='tmp/critic')

    x_sols1, y_sols1, sigma = train(env, x_agent, y_agent, critic, max_games = 500)

    num_walkers = 50000
    num_games = 1

    env = GNEP_ENV_MULTI(obj_func1, obj_func2, x_bounds, y_bounds, constraint_func, num_agents=num_walkers, num_games=num_games)
    x_sols2, y_sols2 = play(env, x_agent, y_agent, num_walkers)

    x_sols2 = np.array(x_sols2)
    y_sols2 = np.array(y_sols2)

    toc = timeit.default_timer()
    time = toc - tic
    print(len(x_sols2))
    print(time)  

    plt.title('Plot for Nash Equilibria')
    #plt.scatter(x_sols1, y_sols1, c='k', marker='.')
    plt.scatter(x_sols2, y_sols2, c='k', marker='.')
    plt.show()
    
#############################################################################################################