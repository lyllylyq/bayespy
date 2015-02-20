import numpy as np
np.random.seed(1)
y0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 0.02]], size=50)
y1 = np.random.multivariate_normal([0, 0], [[0.02, 0], [0, 1]], size=50)
y2 = np.random.multivariate_normal([2, 2], [[1, -0.9], [-0.9, 1]], size=50)
y3 = np.random.multivariate_normal([-2, -2], [[0.1, 0], [0, 0.1]], size=50)
y = np.vstack([y0, y1, y2, y3])

import bayespy.plot as bpplt
bpplt.pyplot.plot(y[:,0], y[:,1], 'rx')
bpplt.pyplot.show()