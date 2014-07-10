from bayespy.nodes import CategoricalMarkovChain
a0 = [0.6, 0.4] # p(rainy)=0.6, p(sunny)=0.4
A = [[0.7, 0.3], # p(rainy->rainy)=0.7, p(rainy->sunny)=0.3
     [0.4, 0.6]] # p(sunny->rainy)=0.4, p(sunny->sunny)=0.6
N = 1000
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
from bayespy.nodes import Dirichlet
a0 = Dirichlet([0.1, 0.1])
A = Dirichlet([[0.1, 0.1],
               [0.1, 0.1]])
Z = CategoricalMarkovChain(a0, A, states=N)
P = Dirichlet([[0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1]])
Y = Mixture(Z, Categorical, P)
Y.observe(activity)
Q = VB(Y, Z, A, a0, P)
P.initialize_from_random()
Q.update(Z, A, a0, P, repeat=20)
import bayespy.plot.plotting as bpplt
bpplt.hinton(P)
bpplt.pyplot.show()