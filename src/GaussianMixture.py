#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from bayespy import nodes
from bayespy.inference import VB

N = 1000
D = 2

data = np.random.randn(N, D)
data[:500] += 5 * np.ones(D)
print(data)
print(data.shape)
plt.scatter(data[:, 0], data[:, 1])
plt.show()

K = 2

mu = nodes.Gaussian(np.zeros(D), 0.01 * np.identity(D), plates=(K,))

Lambda = nodes.Wishart(D, D * np.identity(D), plates=(K,))

alpha = nodes.Dirichlet(0.01 * np.ones(K))
z = nodes.Categorical(alpha, plates=(N,))

x = nodes.Mixture(z, nodes.Gaussian, mu, Lambda)

x.observe(data)

Q = VB(x, mu, z, Lambda, alpha)

z.initialize_from_random()

Q.update(repeat=200)

import bayespy.plot as bpplt

bpplt.gaussian_mixture_2d(x, alpha=alpha)
bpplt.pyplot.show()

print(alpha)
