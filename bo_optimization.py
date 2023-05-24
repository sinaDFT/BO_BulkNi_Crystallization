# Main BO code written by Sina Malakpour Estalaki
# BO works coupled with LAMMPS simulator



import os
from shutil import copyfile
import sys
import numpy as np
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from plot_funcs import plot_acquisition#, plot_convergence
from bayesian_optimization_util import plot_convergence
#from bayesian_optimization_util import plot_approximation, plot_acquisition

def simulator(T_real, P_real, dirc):    
    os.chdir(dirc)
    curdir = os.getcwd()    
    path = dirc+'/'+'T_next'+'_'+str(T_real)+'/'+'P_next'+'_'+str(P_real)+'/'
    try:
        os.makedirs(path, mode = 0o666)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)    
    
    file_path = './input_files/'
    inp1 = os.path.join(file_path, 'runfile_eq')
    inp2 = os.path.join(file_path, 'runfile')
    os.system('cp -r '+file_path+'submit_eq.sh '+path)
    os.system('cp -r '+file_path+'submit.sh '+path)
    out1 = open(inp1).read().format(P_real,P_real,P_real)
    out2 = open(inp2).read().format(P_real,T_real,T_real,T_real,P_real,P_real)
    open(os.path.join(path, 'runfile_eq'), 'w').write(out1)
    open(os.path.join(path, 'runfile'), 'w').write(out2)    
    os.system('cd '+path+'; '+'chmod +x submit_eq.sh submit.sh')
    os.system('cd '+path+'; '+' ./submit_eq.sh')
    os.system('cd '+path+'; '+' ./submit.sh')
    #os.system('cd '+path+'; '+'qsub -sync y submit_eq.sh')
    #os.system('cd '+path+'; '+'qsub -sync y submit.sh')
    curdir = os.getcwd()
    a = os.path.join(curdir, 'T_next'+'_'+str(T_real), 'P_next'+'_'+str(P_real))
    out = os.path.join(a, 'isfcc')
    with open(out) as d:
        content = d.readlines()               
    N = int(content[3])        
    index_first = [x for x in range(len(content)) if 'ITEM: NUMBER OF ATOMS' in content[x]]
    fcc_all = []
    hcp_all = []
    unkwn_all = [] 
    for ind in np.arange(0, len(index_first)):
        h=[]
        fcc=[]
        hcp=[]
        unkwn=[] 
        for i in range(7, N+7):
            h.append(content[index_first[ind] + i].split()[:])
            if h[i-7][1]== '1':
                fcc.append(h[i-7][1])
            elif h[i-7][1]== '2':
                hcp.append(h[i-7][1])
            else:
                unkwn.append(h[i-7][1])    
        fcc_all.append(fcc)
        hcp_all.append(hcp)
        unkwn_all.append(unkwn) 
    n_hcp_atoms = []            
    for i in range(0, len(hcp_all)):    
        n_hcp_atoms.append(len(hcp_all[i]))    
    hcp_atoms_frac = np.array(n_hcp_atoms)/N
    c = []
    for i in range(1, len(hcp_atoms_frac)):
        r = abs(hcp_atoms_frac[i] - hcp_atoms_frac[i-1]) 
        if r <= 0.005:
            c.append(hcp_atoms_frac[i])
    return max(c)    

dat = np.loadtxt("hcp_frac.txt")
T = []
for i in range(3, 21): 
    t = [i*100]*17
    T.append(t)
T = np.array(T)
T = np.reshape(T, (306, -1)) 
p = np.array([1,10,100,500,1000,5000,10000,50000,100000,200000,
              300000,400000,500000,600000,700000,800000,900000])
P = np.tile(p, (18,1))
P = np.reshape(P, (306, -1))
x = np.concatenate((T, P), axis=1)
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
y = np.expand_dims(dat,1)

bounds = np.array([[np.min(x[:,0]), np.max(x[:,0])], [np.min(x[:,1]), np.max(x[:,1])]])
noise = 0.1   
x_init, x_test, y_init, y_test = train_test_split(x, y, test_size=0.9, random_state=239)

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x#.reshape(-1, 1)

# Gaussian process with Matern kernel as surrogate model
Ker = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5)
m52 = Ker + WhiteKernel(noise_level=0.1)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2, random_state=1987)

# Initialize samples
X_sample = x_init
Y_sample = y_init

# Number of iterations
n_iter = 100
n_init = X_sample.shape[0]
#plt.figure(figsize=(12, n_iter * 3))
#plt.subplots_adjust(hspace=0.4)

for i in range(n_iter):
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    x_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)        
    x_h = scaler.inverse_transform(x_next)                
    dirc = "/afs/crc.nd.edu/user/s/smalakpo/A_Ni_Simulations/HCP_A_Ni/BO_model_fullnoise_0.10/" 
    y_next = simulator(x_h[0], x_h[1], dirc)
    print(f'Best_Point {x_h, y_next}')    

    # Plot samples, surrogate function, noise-free objective and next sampling location
    # plt.subplot(n_iter, 2, 2 * i + 1)
    # plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
    # plt.title(f'Iteration {i+1}')

    # plt.subplot(n_iter, 2, 2 * i + 2)
    # plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)    
    # Add sample to previous samples
    X_sample = np.vstack((X_sample, x_next))
    Y_sample = np.vstack((Y_sample, y_next))
    X_real = scaler.inverse_transform(X_sample)
    bounds_real = scaler.inverse_transform(bounds.T)         
    plot_acquisition(bounds, bounds_real, 2, gpr, X_sample, X_real, Y_sample, 
            expected_improvement, x_h, filename=f'Iteration_{i+1}.png', color_by_step=False)

plot_convergence(X_sample, Y_sample, n_init=n_init, filename='Statistics.png')
plot_convergence(X_real, Y_sample, n_init=n_init, filename='Statistics_realfeature.png')                                              