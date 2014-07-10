import numpy
numpy.random.seed(1)
from bayespy.nodes import CategoricalMarkovChain
a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
     [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
N = 100
Z = CategoricalMarkovChain(a0, A, states=N)
from bayespy.nodes import Categorical, Mixture
P = [[0.1, 0.4, 0.5],
     [0.6, 0.3, 0.1]]
Y = Mixture(Z, Categorical, P)
weather = Z.random()
from bayespy.inference import VB
import bayespy.plot as bpplt
import numpy as np
mu = np.array([ [0,0], [3,4], [6,0] ])
std = 2.0
K = 3
N = 200
p0 = np.ones(K) / K
q = 0.9
r = (1-q)/(K-1)
P = q*np.identity(K) + r*(np.ones((3,3))-np.identity(3))
y = np.zeros((N,2))
z = np.zeros(N)
state = np.random.choice(K, p=p0)
for n in range(N):
    z[n] = state
    y[n,:] = std*np.random.randn(2) + mu[state]
    state = np.random.choice(K, p=P[state])
bpplt.pyplot.figure()
bpplt.pyplot.axis('equal')
colors = [ [[1,0,0], [0,1,0], [0,0,1]][int(state)] for state in z ]
bpplt.pyplot.plot(y[:,0], y[:,1], 'k-', zorder=-10)
bpplt.pyplot.scatter(y[:,0], y[:,1], c=colors, s=40)
bpplt.pyplot.show()