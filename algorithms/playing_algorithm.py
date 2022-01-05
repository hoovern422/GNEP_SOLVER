import numpy as np

##############################################################################

def play(env, x_agent, y_agent, num_agents, num_games):

    """
    The play function plays multiple games in parallel using already-trained
    agents.

    :param env: An instance of the PlayingEnv class
    :param x_agent: An instance of the Agent class controlling the x values
    :param y_agent: An instance of the Agent class controlling the y values
    :num_agents: Number of agents playing in parallel
    :num_games: Number of games to be played by all num_agents x and y agent
                pairs

    :return: Two NumPy arrays with the x and y coordinates of each solution
    """

    done = False
    x_sols = []
    y_sols = []
    x_move = True
    y_move = False

    nash_eq_vec = np.zeros(num_agents)
    game_counter_vec = np.zeros(num_agents)

    x_agent.load_models()
    y_agent.load_models()
    env.game_over_vec = np.ones(num_agents)

    sigma = 1e-4

    step_counter = 0

    while not done:

        x_observations, y_observations = env.reset()
        step_counter += 1
        observations = np.stack((x_observations, y_observations), axis=-1)

        if (step_counter % 2 == 0):

            agent = x_agent
            x_move = True
            y_move = False

        else:
            agent = y_agent
            y_move = True
            x_move = False

        actions = agent.choose_actions(observations, sigma)

        x_observations, y_observations, nash_eq_vec, \
            game_counter_vec = env.step(actions, x_move, y_move)

        nash_eq_indices = [i for i, j in enumerate(nash_eq_vec) if j == 1]
        x_sols.extend(x_observations[nash_eq_indices])
        y_sols.extend(y_observations[nash_eq_indices])

        done = (np.sum(game_counter_vec) >= num_agents * num_games)

    return np.array(x_sols), np.array(y_sols)

##############################################################################