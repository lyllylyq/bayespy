import numpy
numpy.random.seed(1)
M = 20
N = 100
import numpy as np
x = np.random.randn(N, 2)
w = np.random.randn(M, 2)
f = np.einsum('ik,jk->ij', w, x)
y = f + 0.1*np.random.randn(M, N)
D = 10
from bayespy.nodes import GaussianARD, Gamma, SumMultiply
X = GaussianARD(0, 1, plates=(1,N), shape=(D,))
alpha = Gamma(1e-5, 1e-5, plates=(D,))
C = GaussianARD(0, alpha, plates=(M,1), shape=(D,))
F = SumMultiply('d,d->', X, C)
tau = Gamma(1e-5, 1e-5)
Y = GaussianARD(F, tau)
Y.observe(y)
from bayespy.inference import VB
Q = VB(Y, X, C, alpha, tau)
C.initialize_from_random()
from bayespy.inference.vmp.transformations import RotateGaussianARD
rot_X = RotateGaussianARD(X)
rot_C = RotateGaussianARD(C, alpha)
from bayespy.inference.vmp.transformations import RotationOptimizer
R = RotationOptimizer(rot_X, rot_C, D)
Q.set_callback(R.rotate)
Q.update(repeat=1000)
import bayespy.plot as bpplt
bpplt.hinton(C)