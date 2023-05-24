# Leave-One-Out Cross validation and GP parity plots


import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor

dat = np.loadtxt("hcp_frac.txt")
T = []
for i in range(3, 19): 
    t = [i*100]*12
    T.append(t)
T = np.array(T)
T = np.reshape(T, (192, -1)) 
p = np.array([10,100,1000,10000,50000,100000,200000,
              400000,500000,700000,800000,900000])
P = np.tile(p, (16,1))
P = np.reshape(P, (192, -1))
x = np.concatenate((T, P), axis=1)
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
y = np.expand_dims(dat,1)
#print(x)
#x_next = [1.63835604, 2.04156057]
#x_h = scaler.inverse_transform(x_next)
#print(x_h)

#print(x.shape)
#print(y.shape)

X, x_test, Y, y_test = train_test_split(x, y, test_size=0.65, random_state=450)
#print(X.shape, Y.shape)
print(X.shape)
print(Y.shape)

noise = 0.03

# Gaussian process with Matern kernel as surrogate model
Ker = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5)
#rbf_ker = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) 
m52 = Ker + WhiteKernel(noise_level=0.03)
#gpr = GaussianProcessRegressor(kernel=m52, random_state=1987)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2, random_state=1987)

loo = LeaveOneOut()
#print(loo.get_n_splits(X))

m_pred = []
std_pred = []
Y_Test = []
for train_index, test_index in loo.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    gpr = gpr.fit(X_train, y_train)
    pred = gpr.predict(X_test, return_std=True)
    m_pred.append(float(pred[0]))
    std_pred.append(float(pred[1]))
    Y_Test.append(float(y_test))

m_pred = np.expand_dims(np.array(m_pred),1)
std_pred = np.array(std_pred)     
Y_Test = np.expand_dims(np.array(Y_Test),1)

dat = np.concatenate((Y_Test,m_pred),1)

fig, ax = plt.subplots()
#ax.errorbar(dat[:,0],dat[:,1],yerr=1.95*np.sqrt(variance),fmt='o',ms=5,color='red',alpha=1.0)
ax.errorbar(dat[:,0],dat[:,1],yerr=std_pred,fmt='o', color='r', 
            markersize=6, markeredgecolor='k', markerfacecolor='r',capsize=2,elinewidth=1.5, mew=2)
#ax.scatter(dat[:,0], dat[:,1], s=40, facecolors='none', edgecolors='b')                
ax.plot([dat[:,0].min(), dat[:,0].max()], [dat[:,0].min(), dat[:,0].max()], 'b--', lw=3)                
ax.set_xlabel('Ground Truth')
ax.set_ylabel('Prediction')
#ax.set_title('LOO')
plt.savefig('GP_67_0.03.png', dpi=600)
    

    
    

