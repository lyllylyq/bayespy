import numpy
numpy.random.seed(1)
import numpy as np
y0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 0.02]], size=50)
y1 = np.random.multivariate_normal([0, 0], [[0.02, 0], [0, 1]], size=50)
y2 = np.random.multivariate_normal([2, 2], [[1, -0.9], [-0.9, 1]], size=50)
y3 = np.random.multivariate_normal([-2, -2], [[0.1, 0], [0, 0.1]], size=50)
y = np.vstack([y0, y1, y2, y3])
import bayespy.plot as bpplt
bpplt.pyplot.plot(y[:,0], y[:,1], 'rx')
N = 200
D = 2
K = 10
from bayespy.nodes import Dirichlet, Categorical
alpha = Dirichlet(1e-5*np.ones(K),
                  name='alpha')
Z = Categorical(alpha,
                plates=(N,),
                name='z')
from bayespy.nodes import Gaussian, Wishart
mu = Gaussian(np.zeros(D), 1e-5*np.identity(D),
              plates=(K,),
              name='mu')
Lambda = Wishart(D, 1e-5*np.identity(D),
                 plates=(K,),
                 name='Lambda')
from bayespy.nodes import Mixture
Y = Mixture(Z, Gaussian, mu, Lambda,
            name='Y')
Z.initialize_from_random()
from bayespy.inference import VB
Q = VB(Y, mu, Lambda, Z, alpha)
Y.observe(y)
Q.update(repeat=1000)

bpplt.gaussian_mixture_2d(Y, alpha=alpha, scale=2)
bpplt.pyplot.show()