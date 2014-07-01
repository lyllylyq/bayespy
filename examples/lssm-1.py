import numpy as np
np.random.seed(1)
from bayespy.nodes import GaussianARD, GaussianMarkovChain, Gamma, Dot
M = 30
N = 400
D = 10
alpha = Gamma(1e-5,
              1e-5,
              plates=(D,),
              name='alpha')
A = GaussianARD(0,
                alpha,
                shape=(D,),
                plates=(D,),
                name='A')
X = GaussianMarkovChain(np.zeros(D),
                        1e-3*np.identity(D),
                        A,
                        np.ones(D),
                        n=N,
                        name='X')
gamma = Gamma(1e-5,
              1e-5,
              plates=(D,),
              name='gamma')
C = GaussianARD(0,
                gamma,
                shape=(D,),
                plates=(M,1),
                name='C')
F = Dot(C,
        X,
        name='F')
C.initialize_from_random()
tau = Gamma(1e-5,
            1e-5,
            name='tau')
Y = GaussianARD(F,
                tau,
                name='Y')
from bayespy.inference import VB
Q = VB(X, C, gamma, A, alpha, tau, Y)
w = 0.3
a = np.array([[np.cos(w), -np.sin(w), 0, 0],
              [np.sin(w), np.cos(w),  0, 0],
              [0,         0,          1, 0],
              [0,         0,          0, 0]])
c = np.random.randn(M,4)
x = np.empty((N,4))
f = np.empty((M,N))
y = np.empty((M,N))
x[0] = 10*np.random.randn(4)
f[:,0] = np.dot(c,x[0])
y[:,0] = f[:,0] + 3*np.random.randn(M)
for n in range(N-1):
    x[n+1] = np.dot(a,x[n]) + [1,1,10,10]*np.random.randn(4)
    f[:,n+1] = np.dot(c,x[n+1])
    y[:,n+1] = f[:,n+1] + 3*np.random.randn(M)
from bayespy.utils import random
mask = random.mask(M, N, p=0.2)
Y.observe(y, mask=mask)
import bayespy.plot as bpplt
X.set_plotter(bpplt.FunctionPlotter(center=True, axis=-2))
A.set_plotter(bpplt.HintonPlotter())
C.set_plotter(bpplt.HintonPlotter())
tau.set_plotter(bpplt.PDFPlotter(np.linspace(0.02, 0.5, num=1000)))
Q.update(repeat=10)
from bayespy.inference.vmp import transformations
rotC = transformations.RotateGaussianARD(C, gamma)
rotA = transformations.RotateGaussianARD(A, alpha)
rotX = transformations.RotateGaussianMarkovChain(X, rotA)
R = transformations.RotationOptimizer(rotX, rotC, D)
Q.callback = R.rotate
Q.update(repeat=1000)
Q.plot(X, A, C, tau)
bpplt.pyplot.show()