#!/usr/bin/env python
# coding: utf-8

# The aim of this script is to test InferPy potential on modelling Gaussian mixtures.
# In this case, Mnist dataset is used as the same outcome happends with both datasets.

# Using this framework, this model cannot be learned due a limitation of
# Tensorflow-Probability where categorical variables cannot be learned using
# a gradient method.

# Imports
from functions import *
from models import mixture, Q_mixture

# Set seed
set_seed(2020)

# Load data from InferPy
(X, y), _ = mnist.load_data(num_instances=100, digits=[1, 2, 3])
# Create dataframe
dataset = pd.DataFrame(data=X)
dataset["number"] = y
# Split dataframe
X = dataset.drop(["number"], axis=1)
y = dataset["number"]

# n_components
n_components = len(y.unique())
# Ammount of samples
n_samples = X.shape[0]
# Observed space dimensionality
observed_space_dim = X.shape[1]

# Show categorial variables are non-reparameterized.
x = inf.Categorical(probs=[0.5, 0.5])
print("Categorical variable reparameterization type: ")
print(x.reparameterization_type)

# Define the model object
model = mixture(n_components, observed_space_dim)

# Define variational model
q = Q_mixture(n_components, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(0.01)
# Define inference object
VI = inf.inference.VI(q, optimizer=optimizer, epochs=2000)

# The model is fitted using the dataset. This raises an error where the concentration parameter of the InverseGamma distribution reaches a non-positive value (even using a softplus function).
#
# This being said, inference is not possible with this model and database.
print("Fitting mixture model:")
model.fit({"x": X}, VI)
