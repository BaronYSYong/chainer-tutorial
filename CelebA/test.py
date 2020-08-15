from __future__ import print_function, unicode_literals

import numpy as np

import chainer
import chainer.functions as F

if __name__ == '__main__':
    a = np.array([[0.0,-0.5],
                  [0.0, 0.5]])
    print(a)
    t = np.array([0,0], 'i')
    tb = np.array([[1,0],
                   [0,1]], 'i')
    print(t)
    print(F.accuracy(a,t).data)
    print(F.binary_accuracy(a,tb).data)

    tc = np.array([[1,0], [1,0], [0,1], [0,1]])
    print(np.where(tc))
    print(np.where(tc)[1])

    x = np.random.random((10,4))
    b1 = (0,2)
    b2 = (2,4)
    print(x)
    print(x[:,slice(*b1)])
    print(x[:,slice(*b2)])
