from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
def bce(o,t):
    return -(np.log(o)*t+np.log(1-o)*(1-t))
def mse(o,t):
    return np.square(o-t)
eps = 1e-3
O = np.linspace(0+eps,1-eps)
T = np.linspace(0+eps,1-eps)
xv,yv = np.meshgrid(O,T)
res = mse(xv,yv)
res2 = bce(xv,yv)
surf = ax.plot_surface(xv,yv,res,cmap=cm.coolwarm)
surf2 = ax.plot_surface(xv,yv,res2,cmap=cm.coolwarm)
#ax.set_zlim(0, 1)
plt.xlabel('output')
plt.ylabel('target')
plt.show()
