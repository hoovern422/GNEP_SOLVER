import numpy as np

##############################################################################

def train(env, x_agent, y_agent, critic, max_games):

    """
    The train function executes the TD(0) actor-critic algorithm employed in
    the training processes of the neural network GNEP solver.

    :param env: An instance of the TrainingEnv class
    :param x_agent: An instance of the Agent class controlling the x values
    :param y_agent: An instance of the Agent class controlling the y values
    :param critic: An instance of the Critic class used as the critic for both
                   agents
    :max_games: Maximum number of games that will be played during training 

    :return: Three NumPy arrays, two with the x and y coordinates of each
             solution found and one with the average rewards from each 
             game
    """

    game_counter = 0
    games_since_last_save = 0
    done = False
    avg_reward = -100
    reward_history = []
    x_sols = []
    y_sols = []
    avg_rewards = []

    while not done:

        # Start a new game.
        game_counter += 1
        games_since_last_save += 1
        step_counter = 0
        state, game_over, is_nash_eq, reward = env.reset()
        x_move = True
        y_move = False
        sigma = 1 / game_counter

        while not game_over:

            # Choose an action for each agent until they are taken outside the
            #  feasible set (defined in the environment).
            step_counter += 1

            if (step_counter % 2 == 0):
                agent = x_agent
                x_move = True
                y_move = False

            else:
                agent = y_agent
                x_move = False
                y_move = True

            action = agent.choose_action(state, sigma)[0][0]

            new_x, new_y, reward, game_over, is_nash_eq \
                = env.step(action, x_move, y_move)

            state_ = [new_x, new_y]
            reward += reward

            critic.learn(state, state_, reward, done)
            state_value = critic.get_prediction(state)
            state_value_ = critic.get_prediction(state_)

            agent.learn(state, reward, done, sigma,
                        state_value, state_value_, action)

            state = state_

        # Find the running average for the rewards from the last 100 games.
        step_counter += (step_counter == 0)*1
        reward = reward / step_counter
        reward_history.append(reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_rewards.append(avg_reward)

        if is_nash_eq:

            x_sols.append(state[0])
            y_sols.append(state[1])
            print(
                'Nash equlibirum point: (',
                state[0],
                ',',
                state[1],
                ')')
            is_nash_eq = False

            if game_counter >= 50:

                x_agent.save_models()
                y_agent.save_models()
                critic.save_models()
                games_since_last_save = 0

        if game_counter == max_games or games_since_last_save >= 100:

            done = True
            x_agent.save_models()
            y_agent.save_models()
            critic.save_models()

        print('Game #', game_counter, 'reward %.3f' % reward,
              'avg_reward %.3f' % avg_reward)

    return np.array(x_sols), np.array(y_sols), np.array(avg_rewards)

##############################################################################