import numpy as np
import matplotlib.pyplot as plt
p = np.load('point_data.npy')
v = np.load('val_data.npy')
x = p[:,0]
y = p[:,1]
mask = v < 1
plt.figure(0)
plt.hist(v[mask])
plt.figure(1)
plt.scatter(x,y,s=100,c=np.log(v))
plt.figure(2)
plt.scatter(x[mask],y[mask],s=100,c=np.log(v[mask]))
plt.show()

