import numpy as np
import scipy as sp
import math
from scipy.optimize import minimize_scalar

class GNEP_ENV(object):

    def __init__(self, obj_func1, obj_func2, x_bounds, y_bounds, constraint_func):

        self.obj_func1 = obj_func1
        self.obj_func2 = obj_func2
        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]
        self.y_min = y_bounds[0]
        self.y_max = y_bounds[1]
        self.constraint_func = constraint_func

        self.cur_x = 0
        self.cur_y = 0
        self.prev_x_opt = 0
        self.prev_y_opt = 0
        self.cur_dist = 1000
        self.prev_dist = 1000
        self.game_over = False
        self.is_nash_eq = False

####################################################################################################################

########################################################################################################################

    def findMaxX(self, y):
        ten_bound = (y < 5)*1
        sum_bound = (y >= 5)*1
        maxX = ten_bound*10 + sum_bound*(15-y)
        return maxX

    def findMaxY(self, x):
        ten_bound = (x < 5)*1
        sum_bound = (x >= 5)*1
        maxY = ten_bound*10 + sum_bound*(15-x) 
        return maxY

########################################################################################################################
    
    # Generate the environment's response to an action taken by the agent.

    def step(self, action, x_move, y_move):

        if x_move:

            self.cur_x = self.cur_x + action
        
        if y_move:

            self.cur_y = self.cur_y + action            

        x_opt = self.optimize_func(self.obj_func1, self.cur_y)
        y_opt = self.optimize_func(self.obj_func2, self.cur_x)
        self.cur_dist = self.get_euclidian_dist(x_opt, y_opt)
        reward = self.get_reward()
        self.prev_dist = self.cur_dist

        if (self.is_valid_move() == False):

            self.game_over = True
            self.is_nash_eq = False

        return self.cur_x, self.cur_y, reward, self.game_over, self.is_nash_eq  

#################################################################################################################### 

    # Determine if the move was valid.

    def is_valid_move(self):

        if (self.cur_x < self.x_min or self.cur_x > self.findMaxX(self.cur_y) \
           or self.cur_y < self.y_min or self.cur_y > self.findMaxY(self.cur_x)):
        
            is_valid = False

        else:

            is_valid = True

        return is_valid    

####################################################################################################################

    def optimize_func(self, func_to_opt, other):

        if (func_to_opt == self.obj_func1):

            lim = self.findMaxX(self.cur_y)

            def func_to_min(x):
                return x**2 + (8/3)*x*other - 34*x

        else:

            lim = self.findMaxY(self.cur_x)

            def func_to_min(y):
                return y**2 + (5/4)*other*y - 24.25*y       

        if (lim < 1.0e-4):
            lim = 1.0e-4

        bounds = (0, lim)

        solver = sp.optimize.minimize_scalar(func_to_min, bounds=bounds, method='bounded', tol=None, options=None)
        opt = solver.x

        return opt
####################################################################################################################        

    def get_euclidian_dist(self, x_opt, y_opt):

        x_dist = abs(self.cur_x - x_opt)
        y_dist = abs(self.cur_y - y_opt)

        return (math.sqrt(x_dist**2 + y_dist**2))    

####################################################################################################################      

    def get_reward(self):

        if (self.cur_dist <= 0.1):

            reward = 0
            self.game_over = True
            self.is_nash_eq = True

        else:

            reward = - self.cur_dist

        return reward    

####################################################################################################################

    def reset(self):

        valid = False

        while (valid == False):

            # Choose a random x coordinate within the constraints
            self.cur_x = (self.x_max - self.x_min) * np.random.random_sample() + self.x_min

            # Choose a random y coordinate within the constrataints.
            self.cur_y = (self.y_max - self.y_min) * np.random.random_sample() + self.y_min

            valid = self.is_valid_move() 

        x_opt = self.optimize_func(self.obj_func1, self.cur_y)
        y_opt = self.optimize_func(self.obj_func2, self.cur_x)    
        self.prev_dist = self.get_euclidian_dist(x_opt, y_opt)         

        self.game_over = False
        self.is_nash_eq = False

        new_observation = [self.cur_x, self.cur_y]

        return new_observation

####################################################################################################################  