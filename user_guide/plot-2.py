# This is the PCA model from the previous sections
import numpy as np
np.random.seed(1)
from bayespy.nodes import GaussianARD, Gamma, Dot
D = 3
X = GaussianARD(0, 1,
                shape=(D,),
                plates=(1,100),
                name='X')
alpha = Gamma(1e-3, 1e-3,
              plates=(D,),
              name='alpha')
C = GaussianARD(0, alpha,
                shape=(D,),
                plates=(10,1),
                name='C')
F = Dot(C, X)
tau = Gamma(1e-3, 1e-3, name='tau')
Y = GaussianARD(F, tau)
c = np.random.randn(10, 2)
x = np.random.randn(2, 100)
data = np.dot(c, x) + 0.1*np.random.randn(10, 100)
Y.observe(data)
Y.observe(data, mask=[[True], [False], [False], [True], [True],
                      [False], [True], [True], [True], [False]])
from bayespy.inference import VB
Q = VB(Y, C, X, alpha, tau)
X.initialize_from_parameters(np.random.randn(1, 100, D), 10)
from bayespy.inference.vmp import transformations
rotX = transformations.RotateGaussianARD(X)
rotC = transformations.RotateGaussianARD(C, alpha)
R = transformations.RotationOptimizer(rotC, rotX, D)
Q = VB(Y, C, X, alpha, tau)
Q.callback = R.rotate
Q.update(repeat=1000, tol=1e-6)
Q.update(repeat=50, tol=np.nan)

import bayespy.plot as bpplt
from bayespy.nodes import Gaussian
V = Gaussian([3, 5], [[4, 2], [2, 5]])
bpplt.contour(V, np.linspace(1, 5, num=100), np.linspace(3, 7, num=100))