import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

iters = int(sys.argv[1])
x = np.loadtxt('iter_' + str(iters) + '_mode_2_pts_Y_70000', delimiter=' ')

mpl.rcParams['agg.path.chunksize'] = 10000
plt.figure(figsize=(10,10), dpi=600)
plt.scatter(x[:,0], x[:,1], marker=',', s=72./600)
plt.axis('off')
plt.savefig('res_mnist_' + str(iters) + '.png')




