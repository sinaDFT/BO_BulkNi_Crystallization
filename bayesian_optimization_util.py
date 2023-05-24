import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab
     
def plot_convergence(X_sample, Y_sample, n_init=2, filename=None):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:, :]
    n = x.shape[0]
    aux = (x[1:n,:]-x[0:n-1,:])**2
    distances = np.sqrt(aux.sum(axis=1))
    y = Y_sample[n_init:].ravel()
    #r = range(1, len(x)+1)
    
    #x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)
    
    ## Distances between consecutive x's
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n-1)), distances, '-ro')
    plt.xlabel('Iteration', fontsize=18.0, fontweight='bold')
    plt.ylabel('d(x[n], x[n-1])', fontsize=18.0, fontweight='bold')
    plt.title('Distance between consecutive x\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)),y_max_watermark,'-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration', fontsize=18.0, fontweight='bold')
    plt.ylabel('Best y', fontsize=18.0, fontweight='bold')
    grid(True)

    if filename!=None:
        savefig(filename, dpi=800)
    else:
        plt.show()