import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

##############################################################################

def plot_learning_curve(episode_tracker, scores, problem):

    """
    Plots a learning curve. 
    """

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(episode_tracker, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig('plots/' + problem + '/learning_curve.png')
    plt.show()           

##############################################################################

def plot_nash_eq(x_sols, y_sols, problem):

    """
    Creates a scatterplot of Nash equilibiria.
    """

    plt.title('Plot for Nash Equilibria (Playing)')
    plt.scatter(x_sols, y_sols, c='k', marker='.')
    plt.savefig('plots/' + problem + '/nash_eq.png')
    plt.show()

##############################################################################

def plot_heatmap(x_sols, y_sols, problem):

    """
    Produces a heatmap of the Nash equilibria found.
    """

    data = np.vstack([x_sols, y_sols])
    z = gaussian_kde(data)(data)
    idx = z.argsort()
    x_sols, y_sols, z = x_sols[idx], y_sols[idx], z[idx]
    _, ax = plt.subplots()
    cax = ax.scatter(x_sols, y_sols, c=z, s=100)
    plt.colorbar(cax)
    plt.title("Heatmap ")
    plt.savefig('plots/' + problem + '/heatmap.png')
    plt.show()        

##############################################################################