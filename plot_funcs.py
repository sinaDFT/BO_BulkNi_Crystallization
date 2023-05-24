# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license
# Some changes to the code done by Sina Malakpour Estalaki

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab


def plot_acquisition(bounds, bounds_real, input_dim, model, Xdata, Xreal, Ydata, acquisition_function, suggested_sample,
                     filename=None, label_x=None, label_y=None, color_by_step=True):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:

        if not label_x:
            label_x = 'x'

        if not label_y:
            label_y = 'f(x)'

        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(x_grid)


        model.plot_density(bounds[0], alpha=.5)

        plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
        plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
        plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

        plt.plot(Xdata, Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

        plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')


        if filename!=None:
            savefig(filename)
        else:
            plt.show()

    if input_dim == 2:

        if not label_x:
            label_x = 'T'

        if not label_y:
            label_y = 'P'

        n = Xdata.shape[0]
        colors = np.linspace(0, 1, n)
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=1)
        points_var_color = lambda X: plt.scatter(
            X[:,0], X[:,1], c=colors, label=u'Observations', cmap=cmap, norm=norm)
        points_one_color = lambda X: plt.plot(
            X[:,0], X[:,1], 'r.', markersize=10, label=u'Observations')
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X, Xdata, Ydata, model)
        #acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = (acqu - min(acqu))/(max(acqu - min(acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X, return_std=True)
        X1_real = np.linspace(bounds_real.T[0][0], bounds_real.T[0][1], 200)
        X2_real = np.linspace(bounds_real.T[1][0], bounds_real.T[1][1], 200)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1_real, X2_real, m.reshape(200,200),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xreal)
        else:
            points_one_color(Xreal)
        #plt.ylabel(label_y)
        plt.title('Posterior mean')
        plt.axis((bounds_real.T[0][0],bounds_real.T[0][1],bounds_real.T[1][0],bounds_real.T[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.contourf(X1_real, X2_real, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xreal)
        else:
            points_one_color(Xreal)
        #plt.xlabel(label_x)
        #plt.ylabel(label_y)
        plt.title('Posterior sd.')
        plt.axis((bounds_real.T[0][0],bounds_real.T[0][1],bounds_real.T[1][0],bounds_real.T[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1_real, X2_real, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[0],suggested_sample[1],'m.', markersize=10)
        #plt.xlabel(label_x)
        #plt.ylabel(label_y)
        plt.title('Acquisition function')
        plt.axis((bounds_real.T[0][0],bounds_real.T[0][1],bounds_real.T[1][0],bounds_real.T[1][1]))
        if filename!=None:
            plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.6)
            savefig(filename, dpi=800)
        else:
            plt.show()    