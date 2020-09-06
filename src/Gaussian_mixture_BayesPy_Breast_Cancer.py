#!/usr/bin/env python
# coding: utf-8

# The aim of this script is to test BayesPy potential on modelling Gaussian mixtures.
# Two attepts are done, the first on the proper Breast Cancer Wisconsin dataset,
# and the second on a two-dimensional reduction using a variational auto-encoder.

# The mixture is componed by the same number of components as classes has the dataset.
# An optimal outcome would be that each class is modeled by one component.
# In order to test this, we are using each datapoint posterior probability of belonging
# to each component to compute a density function over each class and component.

# Imports
from functions import *
from models import *
from bayespy import nodes
from bayespy.inference import VB

# seed
set_seed(2020)

# Load and prepare Breast Cancer database
# Column names
col_names = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]
# Reading dataset
dataset = pd.read_csv("../dataset/wdbc.data", header=None, names=col_names)
X = dataset.drop(["id", "diagnosis"], axis=1)
y = dataset["diagnosis"]

# Ammount of samples
n_samples = X.shape[0]
# Observed space dimensionality
dim = X.shape[1]
# Ammount of classes
n_classes = len(y.unique())


# Define a function that returns all the variables of the model.
def GaussianMixture(dim, n_components, n_samples):
    """
    Mean and precision distribucion defined as
    $$
      \mu \sim \mathcal{N}_{dim}(0,I) \quad \text{and} \quad \Lambda \sim \mathcal{W}(dim,I_{dim}).
    $$
    Mixture weights follow a Dirichlet distribution
    $$
      \pi \sim \text{Symmetric-Dirichlet}\Big(\frac{1}{n\_components}\Big).
    $$
    The variable modeling the component assignation follows a categorical distribution as
    $$
      Z_n \sim \text{Categorical}(\pi).
    $$
    """
    Lambda = nodes.Wishart(dim, np.identity(dim), plates=(n_components,))
    mu = nodes.Gaussian(np.zeros(dim), np.identity(dim), plates=(n_components,))
    pi = nodes.Dirichlet(np.ones(n_components) / n_components)
    z = nodes.Categorical(pi, plates=(n_samples,))
    x = nodes.Mixture(z, nodes.Gaussian, mu, Lambda)

    return x, z, pi, mu, Lambda


# Define the model.
x, z, pi, mu, Lambda = GaussianMixture(
    dim=X.shape[1], n_components=len(y.unique()), n_samples=X.shape[0]
)

# Pandas' dataframes can't be used. to observe the data as they raise a dimensionality error.
# We may transform it to a Numpy array
x.observe(X.to_numpy())

# Create the inference object.
Q = VB(x, mu, z, Lambda, pi)

# Initialize z using random values as letting the default initialization
# (prior) leads to the same 2 distributions.
z.initialize_from_random()

# Start learning.
print("Variational message passing iterations:")
Q.update(repeat=1000)

# Each component probability may be inspected using `pi`.
print("Learned weights:")
print(pi)
input("Press Enter to continue...")

# `Z` models each datapoint component and its first moment shows the probability
# of belonging to each component.
print("Probabilities of benign cases to belong to each component:")
print(z.u[0][y == "B"])
input("Press Enter to continue...")

# Plot each component belonging density.
print("Component belonging densities (image).")
plot_mixture_distplot(
    [z.u[0][y == "B"], z.u[0][y == "M"]],
    ["Benign", "Malign"],
    ["blue", "green"],
    n_components=n_classes,
)

# We might generate a posterior sample using `random()`.
print("Posterior sample:")
print(x.random())
input("Press Enter to continue...")

# We might use a variational auto-encoder to reduce the data to a two-dimensional space.
vae_model = vae(2, 100, dim)
q_vae = Q_vae(2, 100, dim)

optimizer = tf.train.AdamOptimizer(0.01)
VI_vae = inf.inference.VI(q_vae, optimizer=optimizer, epochs=4000)
print("VAE model fitting:")
vae_model.fit({"x": X}, VI_vae)
data = vae_model.posterior("z", data={"x": X}).sample()

# We define the new mixture model and initialize training.
x, z, pi, mu, Lambda = GaussianMixture(
    dim=data.shape[1], n_components=len(y.unique()), n_samples=X.shape[0]
)
x.observe(data)
Q = VB(x, mu, z, Lambda, pi)
z.initialize_from_random()
print("Variational message passing iterations.")
Q.update(repeat=1000)

# In this case, `pi` has correctly learn each component probability.
print("\nLearned weights:")
print(pi)

# `BayesPy` has several plotting functions, among them, `gaussian_mixture_2d` should plot all datapoints and Gaussian ellipses. For some reason, it fails to plot both ellipses.
bpplt.gaussian_mixture_2d(x, alpha=pi, scale=5)
print("Plot datapoints and Gaussian ellipses (image).")
bpplt.pyplot.show()

# Component belonging densities are shown.
print("Component belonging densities (image).")
plot_mixture_distplot(
    [z.u[0][y == "B"], z.u[0][y == "M"]],
    ["Benign", "Malign"],
    ["blue", "green"],
    n_components=n_classes,
)
