import numpy as np
import shelve
from BPSA import *


samples = [
        (np.mat([5.33, 5.39, 5.29, 5.41, 5.45]), 5.50),
        (np.mat([5.39, 5.29, 5.41, 5.45, 5.50]), 5.57),
        (np.mat([5.29, 5.41, 5.45, 5.50, 5.57]), 5.58),
        (np.mat([5.41, 5.45, 5.50, 5.57, 5.58]), 5.61),
        (np.mat([5.45, 5.50, 5.57, 5.58, 5.61]), 5.69),
        (np.mat([5.50, 5.57, 5.58, 5.61, 5.69]), 5.78),
        (np.mat([5.57, 5.58, 5.61, 5.69, 5.78]), 5.78),
        (np.mat([5.58, 5.61, 5.69, 5.78, 5.78]), 5.81),
        (np.mat([5.61, 5.69, 5.78, 5.78, 5.81]), 5.86),
        (np.mat([5.69, 5.78, 5.78, 5.81, 5.86]), 5.90),
        (np.mat([5.78, 5.78, 5.81, 5.86, 5.90]), 5.97),
        (np.mat([5.78, 5.81, 5.86, 5.90, 5.97]), 6.49),
        (np.mat([5.81, 5.86, 5.90, 5.97, 6.49]), 6.60),
        (np.mat([5.86, 5.90, 5.97, 6.49, 6.60]), 6.64)
    ]
tests = [
    (np.mat([5.90, 5.97, 6.49, 6.60, 6.64]), 6.74),
    (np.mat([5.97, 6.49, 6.60, 6.64, 6.74]), 6.87),
    (np.mat([6.49, 6.60, 6.64, 6.74, 6.87]), 7.01)
]

w1 = np.random.rand(5, 8)*10-5
theta1 = np.random.rand(1, 8)*10-5
W1 = np.asmatrix(np.vstack([theta1,w1]))
w2 = np.random.rand(8, 1)*10-5
theta2 = np.random.rand(1, 1)*10-5
W2 = np.asmatrix(np.vstack([theta2, w2]))
W = [W1, W2]


with open('./t.txt', 'r') as f:
    x = f.read()
    if x == '':
        i = 0
    else:
        i = int(x)
i = i + 1
with open('./t.txt', 'w') as f:
    f.write(str(i))

W = BPSA.train(samples, W, fname=str(i),tests=tests)
