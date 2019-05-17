import pandas as pd
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D

def optimization_loss_plot(path='bayesian_parameter_estimation.log'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    log_data = pd.read_csv(path, sep=',', engine='python')

    ax.scatter(log_data.interm, log_data.latent, zs=log_data.target)
    ax.set_xlabel('intermediate dimensions')
    ax.set_ylabel('latent dimensions')
    ax.set_zlabel('loss')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[-1]
        optimization_loss_plot(file_name)
    else:
        optimization_loss_plot()