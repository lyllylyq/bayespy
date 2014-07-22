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
activity = Mixture(weather, Categorical, P).random()
Y.observe(activity)
from bayespy.inference import VB
Q = VB(Y, Z)
Q.update()
import bayespy.plot as bpplt
bpplt.plot(Z)
bpplt.plot(1-weather, color='r', marker='x')
bpplt.pyplot.show()