from GNEP_environments.env4.optimizers import optimize_x, optimize_y, find_max_x, find_max_y
import math
import numpy as np

##############################################################################

class TrainingEnv(object):

    """ 
    The environment used when training the actor and critic networks in the 
    GNEP solver.
    """
    def __init__(self, x_bounds, y_bounds, threshold, reward_scaling_factor):

        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]
        self.y_min = y_bounds[0]
        self.y_max = y_bounds[1]
        self.threshold = threshold
        self.reward_scaling_factor = reward_scaling_factor

        self.cur_x = 0
        self.cur_y = 0
        self.prev_x_opt = 0
        self.prev_y_opt = 0
        self.cur_dist = 1000
        self.game_over = False
        self.is_nash_eq = False

##############################################################################

    def step(self, action, x_move, y_move):

        """
        Generate the environment's response to an action taken by the agent.
        """

        if x_move:

            self.cur_x = self.cur_x + action

        if y_move:

            self.cur_y = self.cur_y + action

        x_opt = optimize_x(self.cur_y)
        y_opt = optimize_y(self.cur_x)
        self.cur_dist = self.get_euclidian_dist(x_opt, y_opt)
        reward = self.get_reward()

        if reward == 0 :

            self.game_over = True
            self.is_nash_eq = True

        if not self.is_valid_move() and not self.is_nash_eq:

            self.game_over = True
            self.is_nash_eq = False    

        return self.cur_x, self.cur_y, reward, self.game_over, self.is_nash_eq

##############################################################################

    def is_valid_move(self):

        """
        Determine if the move was valid.
        """

        if (self.cur_x < self.x_min or self.cur_x > find_max_x(self.cur_y)
           or self.cur_y < self.y_min or self.cur_y > find_max_y(self.cur_x)):

            is_valid = False

        else:

            is_valid = True

        return is_valid

##############################################################################

    def get_euclidian_dist(self, x_opt, y_opt):


        """
        Get the Euclidian distance between two points
        """

        x_dist = abs(self.cur_x - x_opt)
        y_dist = abs(self.cur_y - y_opt)

        return (math.sqrt(x_dist**2 + y_dist**2))

##############################################################################

    def get_reward(self):

        """
        Get the reward based on the current location's distance from the
        shadow point.
        """

        if (self.cur_dist <= self.threshold):

            reward = 0

        else:

            reward = - self.cur_dist * self.reward_scaling_factor

        return reward

##############################################################################

    def reset(self):

        """"
        Reset a game.
        """

        valid = False
        self.game_over = False
        self.is_nash_eq = False

        while not valid:

            # Choose a random x coordinate within the constraints.
            self.cur_x = (self.x_max - self.x_min) * \
                np.random.random_sample() + self.x_min

            # Choose a random y coordinate within the constrataints.
            self.cur_y = (self.y_max - self.y_min) * \
                np.random.random_sample() + self.y_min

            valid = self.is_valid_move()

        _, _, reward, self.game_over, self.is_nash_eq = \
        self.step(0, False, False)

        new_state = [self.cur_x, self.cur_y]

        return new_state, self.game_over, self.is_nash_eq, reward

##############################################################################