import numpy
numpy.random.seed(1)
k = 2 # slope
c = 5 # bias
s = 2 # noise standard deviation
import numpy as np
x = np.arange(10)
y = k*x + c + s*np.random.randn(10)
X = np.vstack([x, np.ones(len(x))]).T
from bayespy.nodes import GaussianARD
B = GaussianARD(0, 1e-6, shape=(2,))
from bayespy.nodes import SumMultiply
F = SumMultiply('i,i', B, X)
from bayespy.nodes import Gamma
tau = Gamma(1e-3, 1e-3)
Y = GaussianARD(F, tau)
Y.observe(y)
from bayespy.inference import VB
Q = VB(Y, B, tau)
Q.update(repeat=1000)
xh = np.linspace(-5, 15, 100)
Xh = np.vstack([xh, np.ones(len(xh))]).T
Fh = SumMultiply('i,i', B, Xh)
import bayespy.plot as bpplt
bpplt.pdf(tau, np.linspace(1e-6,1,100), color='k')
bpplt.pyplot.axvline(s**(-2), color='r')
bpplt.pyplot.show()