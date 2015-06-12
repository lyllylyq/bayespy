#!/usr/bin/env python
# coding: utf-8
"""
Project:  bayespy
Title:    regression 
Author:   Liuyl 
DateTime: 2015/6/12 14:10 
UpdateLog:
1、Liuyl 2015/6/12 Create this File.

regression
>>> print("No Test")
No Test
"""
__author__ = 'Liuyl'
import numpy as np
import bayespy.plot as bpplt
from bayespy.nodes import GaussianARD
from bayespy.nodes import SumMultiply
from bayespy.nodes import Gamma
def run():
    k = 2
    c = 5
    s = 2
    x = np.arange(10)
    y = k * x + c + s * np.random.randn(10)

    X=np.vstack([x,np.ones(len(x))]).T
    B = GaussianARD(0, 1e-6, shape=(2,))
    F = SumMultiply('i,i', B, X)
    tau = Gamma(1e-3, 1e-3)

    Y = GaussianARD(F, tau)

    Y.observe(y)
    from bayespy.inference import VB
    Q = VB(Y, B, tau)
    Q.update(repeat=1000)
    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([xh, np.ones(len(xh))]).T
    Fh = SumMultiply('i,i', B, Xh)


    bpplt.pyplot.figure()
    bpplt.plot(Fh, x=xh, scale=2)
    bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
    bpplt.plot(k*xh+c, x=xh, color='r');
    bpplt.pyplot.show()


if __name__ == '__main__':
    run()