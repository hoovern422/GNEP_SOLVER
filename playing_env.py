import numpy as np

class GNEP_ENV_MULTI(object):

    def __init__(self, obj_func1, obj_func2, x_bounds, y_bounds, constraint_func, num_agents, num_games):

        self.obj_func1 = obj_func1
        self.obj_func2 = obj_func2
        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]
        self.y_min = y_bounds[0]
        self.y_max = y_bounds[1]
        self.constraint_func = constraint_func
        self.num_agents = num_agents
        self.num_games = num_games

        self.x_positions = np.zeros(self.num_agents)
        self.y_positions = np.zeros(self.num_agents)
        self.distances = np.zeros(self.num_agents)
        self.rewards = np.zeros(self.num_agents)
        self.nash_eq_vec = np.zeros(self.num_agents)
        self.game_over_vec = np.zeros(self.num_agents)
        self.game_counter_vec = np.zeros(self.num_agents)

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

    def optimizeX(self, y):

        upperLim = self.findMaxX(y)
        lowerLim = np.zeros(self.num_agents)

        def funcx(x):
            return x**2 + (8/3)*x*y - 34*x

        myOpt = self.golden_vec_search(lowerLim, upperLim, funcx)

        return myOpt

    def optimizeY(self, x):

        upperLim = self.findMaxY(x)
        lowerLim = np.zeros(self.num_agents)

        def funcy(y):
           return y**2 + (5/4)*x*y - 24.25*y

        myOpt = self.golden_vec_search(lowerLim, upperLim, funcy)

        return myOpt        

########################################################################################################################

    def step(self, actions, x_move, y_move):

        self.x_positions += actions * x_move
        self.y_positions += actions * y_move        

        opt_x_vec = self.optimizeX(self.y_positions)
        opt_y_vec = self.optimizeY(self.x_positions)

        self.distances = self.get_distances(opt_x_vec, opt_y_vec)

        is_nash_eq = np.less_equal(self.distances, 0.1) * 1
        self.nash_eq_vec += is_nash_eq
        self.game_over_vec += is_nash_eq

        is_game_over = np.less(self.x_positions, self.x_min) * 1
        self.game_over_vec += is_game_over

        max_x_vec = self.findMaxX(self.y_positions)
        game_over_indices = [i for i,j in enumerate(self.x_positions) if j > max_x_vec[i]]
        self.game_over_vec[game_over_indices] = 1
        #is_game_over = np.greater(self.x_positions, self.x_max) * 1
        #self.game_over_vec += is_game_over

        is_game_over = np.less(self.y_positions, self.y_min) * 1
        self.game_over_vec += is_game_over

        max_y_vec = self.findMaxY(self.x_positions)
        game_over_indices = [i for i,j in enumerate(self.y_positions) if j > max_y_vec[i]]
        self.game_over_vec[game_over_indices] = 1

        #is_game_over = np.greater(self.y_positions, self.y_max) * 1
        #self.game_over_vec += is_game_over

        self.game_counter_vec += self.game_over_vec

        return self.x_positions, self.y_positions, self.nash_eq_vec, self.game_counter_vec

########################################################################################################################

    def golden_vec_search(self, lowB, upB, f):
        R = (5**0.5 - 1) / 2
        D = R * (upB - lowB)
        x1 = lowB + D
        x2 = upB - D
        f1 = f(x1)
        f2 = f(x2)
        
        for i  in range(50):
        #while max(abs(f1-f2)) > 0.0001:
            f1Lower = (f1 < f2)*1
            f2Lower = (f1Lower - 1)*(-1)
            
            lowB = x2*f1Lower + lowB*f2Lower
            x2 = x1* f1Lower + x2*f2Lower
            f2 = f1*f1Lower + f2*f2Lower
            x1 = (lowB + R*(upB-lowB))*f1Lower + x1*f2Lower
            f1 = f(x1)*f1Lower + f1*f2Lower

            upB = x1*f2Lower + upB*f1Lower
            x1 = x2*f2Lower + x1*f1Lower
            f1 = f2*f2Lower + f1*f1Lower
            x2 = (upB - R*(upB-lowB))*f2Lower + x2*f1Lower
            f2 = f(x2)*f2Lower + f2*f1Lower

        f1Lower = (f1 < f2)*1
        f2Lower = (f1Lower - 1)*(-1)
        xOpt = x1*f1Lower + x2*f2Lower

        return xOpt

####################################################################################################################    

    def get_distances(self, opt_x_vec, opt_y_vec):

        x_distances = self.x_positions - opt_x_vec
        x_distances = abs(x_distances)
        y_distances = self.y_positions - opt_y_vec
        y_distances = abs(y_distances)

        distances = x_distances**2 + y_distances**2
        distances = np.sqrt(distances)

        return distances      

#####################################################################################################################

    def remove_finished_players(self):

        game_limit_vec = (np.array(self.game_counter_vec) >= self.num_games) * 1
        done_playing = [i for i,j in enumerate(game_limit_vec) if j == 1]

        self.x_positions = np.delete(self.x_positions, done_playing)
        self.y_positions = np.delete(self.y_positions, done_playing)
        self.game_over_vec = np.delete(self.game_over_vec, done_playing)
        self.nash_eq_vec = np.delete(self.nash_eq_vec, done_playing)
        self.game_counter_vec = np.delete(self.game_counter_vec, done_playing)
        self.distances = np.delete(self.distances, done_playing)
        self.rewards = np.delete(self.rewards, done_playing)
        self.num_agents = self.num_agents - len(done_playing)

        done = (self.num_agents == 0)

        return done, self.num_agents        

#####################################################################################################################

    def reset(self):

        game_over_vec = np.equal(self.game_over_vec, 1) * 1

        reset_x_vec = game_over_vec * (-1) * self.x_positions
        self.x_positions += reset_x_vec

        reset_y_vec = game_over_vec * (-1) * self.y_positions
        self.y_positions += reset_y_vec

        self.x_positions += ((self.x_max - self.x_min) * np.random.random_sample(self.num_agents) + self.x_min) * self.game_over_vec
        self.y_positions += ((self.findMaxY(self.x_positions) - self.y_min) * np.random.random_sample(self.num_agents) + self.y_min) * self.game_over_vec
        
        self.game_over_vec[:] = 0
        self.nash_eq_vec[:] = 0

        return self.x_positions, self.y_positions

#####################################################################################################################        