from GNEP_environments.env1.optimizers import parallel_optimize_x, parallel_optimize_y, find_max_x, find_max_y
import numpy as np

##############################################################################

class PlayingEnv(object):

    """ 
    The environment used when playing games in parallel using trained agents.
    """

    def __init__(self, x_bounds, y_bounds, num_agents, num_games, threshold):

        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]
        self.y_min = y_bounds[0]
        self.y_max = y_bounds[1]
        self.num_agents = num_agents
        self.num_games = num_games
        self.threshold = threshold

        self.x_positions = np.zeros(self.num_agents)
        self.y_positions = np.zeros(self.num_agents)
        self.distances = np.zeros(self.num_agents)
        self.nash_eq_vec = np.zeros(self.num_agents)
        self.game_over_vec = np.zeros(self.num_agents)
        self.game_counter_vec = np.zeros(self.num_agents)      

##############################################################################

    def step(self, actions, x_move, y_move):

        """ 
        Generates the environment's response to the actions taken by x and y
        agents.
       """

        self.x_positions += actions * x_move
        self.y_positions += actions * y_move        

        opt_x_vec = parallel_optimize_x(self.y_positions, self.num_agents)
        opt_y_vec = parallel_optimize_y(self.x_positions, self.num_agents)

        self.distances = self.get_distances(opt_x_vec, opt_y_vec)

        is_nash_eq = np.less_equal(self.distances, self.threshold) * 1
        self.nash_eq_vec += is_nash_eq
        self.game_over_vec += is_nash_eq

        is_game_over = np.less(self.x_positions, self.x_min) * 1
        self.game_over_vec += is_game_over

        max_x_vec = find_max_x(self.y_positions)
        game_over_indx = \
        [i for i,j in enumerate(self.x_positions) if j > max_x_vec[i]]
        self.game_over_vec[game_over_indx] = 1

        is_game_over = np.less(self.y_positions, self.y_min) * 1
        self.game_over_vec += is_game_over

        max_y_vec = find_max_y(self.x_positions)
        game_over_indx = \
        [i for i,j in enumerate(self.y_positions) if j > max_y_vec[i]]
        self.game_over_vec[game_over_indx] = 1

        self.game_counter_vec += self.game_over_vec

        return self.x_positions, self.y_positions, self.nash_eq_vec, \
               self.game_counter_vec

##############################################################################

    def get_distances(self, opt_x_vec, opt_y_vec):

        """
        Calculates the Euclidian distances between the NumPy arrays opt_x_vec
        and opt_y_vec.
        """

        x_distances = self.x_positions - opt_x_vec
        x_distances = abs(x_distances)
        y_distances = self.y_positions - opt_y_vec
        y_distances = abs(y_distances)

        distances = x_distances**2 + y_distances**2
        distances = np.sqrt(distances)

        return distances      

##############################################################################

    def reset(self):

        """
        Reset all ongoing games.
        """

        game_over_vec = np.equal(self.game_over_vec, 1) * 1

        reset_x_vec = game_over_vec * (-1) * self.x_positions
        self.x_positions += reset_x_vec

        reset_y_vec = game_over_vec * (-1) * self.y_positions
        self.y_positions += reset_y_vec

        self.x_positions += \
        ((self.x_max - self.x_min) \
        * np.random.random_sample(self.num_agents) + self.x_min) \
        * self.game_over_vec

        self.y_positions += \
        ((find_max_y(self.x_positions) - self.y_min) \
        * np.random.random_sample(self.num_agents) + self.y_min) \
        * self.game_over_vec
        
        self.game_over_vec[:] = 0
        self.nash_eq_vec[:] = 0

        return self.x_positions, self.y_positions

##############################################################################