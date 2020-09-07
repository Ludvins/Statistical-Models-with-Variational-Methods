#!/usr/bin/env python
# coding: utf-8

# The aim of this script is to test the usage of PCA, NLPCA and VAE in
# `Breast Cancer Wisconsin` database using`InferPy`.

# All three models are used over two and three dimensional reductions,
# showing all possible 2D projections of the data and the evolution of the loss function
# (-ELBO) in the two-dimensional ones.

# After each reduction, the separability of the reduction is tested using a support
# vector machine, this is made to compare whether the reducted space preserves
# class separability.

# Import auxiliary functions and models
from functions import *
from models import *

# Set seed in all packages
set_seed(2020)

# Firsly, the dataset is read using Pandas. The file is supposed to be located in ```../dataset/```.
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

# Display the first 5 rows of the dataset.
dataset.head()

# We separate the diagnosis column and drop the identifier (non-predictive variable).
X = dataset.drop(["id", "diagnosis"], axis=1)
y = dataset["diagnosis"]

# We shown the number of Benign and Malign samples.
print("Class proportion diagram (image)")
print_class_pie_diagram(y, ["Benign", "Malign"])

# Hidden space dimensionality
hidden_space_dim = 2
# Ammount of samples
n_samples = X.shape[0]
# Observed space dimensionality
observed_space_dim = X.shape[1]
# Ammount of classes
n_classes = len(y.unique())
# Hidden layer dimension for Non-linear PCA and VAE
hidden_layer_dim = 100
# Training epochs
num_epochs = 4000
# Learning rate for Adam optimizer
learning_rate = 0.01

##############################################
###### Reduction to 2-dimensional space ######
##############################################
print(" -> 2-Dimensional Results <- ")

###### Probabilistic PCA
"""
The data `X` is generated assuming that both the hidden representation `Z` and the transformation `W` follow standard Gaussian distributions:
$$
  Z \sim \mathcal{N}_{hidden\_space\_dim}(0,I), \quad W \sim \mathcal{N}_{hidden\_space\_dim \times observed\_space\_dim}(0, I).
$$
A third variable $\delta$ is used to generate non-centered points.
$$
  \delta \sim \mathcal{N}_{observed\_space\_dim}(0,I)
$$
The observed data follows a Gaussian distribution as:
$$
  X \mid z, w, \delta \sim \mathcal{N}_{observed\_space\_dim}(w^T z + \delta, I).
$$
"""

# create an instance of the P model and the Q model
pca_model = pca(hidden_space_dim, observed_space_dim)
pca_q = Q_pca(hidden_space_dim, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)

# Define inference object
VI_pca = inf.inference.VI(pca_q, optimizer=optimizer, epochs=num_epochs)

# We fit the model using the dataset ```X```.
print("Fitting PCA model:")
pca_model.fit({"x": X}, VI_pca)

# The evolution of the loss function (-ELBO) over the training is may be printed using `print_loss_function`.
print("\nPCA loss function evolution (image).")
print_loss_function(VI_pca.losses)

# A posterior sample can be generated using the given dataset. This posterior points and densities can be ploted using `print_posterior`.
z = pca_model.posterior("z", data={"x": X}).sample()
print("PCA posterior sample (image).")
print_posterior(z, y)

# Posterior information can be obtained by inspecting each variable's parameter. For example, location and scale from `W` can be seen as:
post = pca_model.posterior("w").parameters()
print("Variable W posterior parameters:")
print("Loc:", post["loc"])
print("Scale:", post["scale"])

# An SVM is trained over the hidden representation in order to test separability.
print("Separability results:")
test_separability(X, z, y)


####### Non-linear PCA
"""
The data `X` is generated supposing its hidden representation `Z` follow a standard Gaussian distribution:
$$
  Z \sim \mathcal{N}_{hidden\_space\_dim}(0,I).
$$
A two layer network is used, made of four variables following a standard Gaussian distribution, $\alpha_0, \alpha_1, \beta_0$ and $\beta_1$. Only one activation function is considered, as the output is used for a mean value.
$$
  f(z) = (relu(z^T \beta_0 + \alpha_0))^T\beta_1 + \alpha_1.
$$
The observed data follows a Gaussian distribution using $f$ for its mean value:
$$
  X \mid z \sim \mathcal{N}_{observed\_space\_dim}(f(z), I).
$$
"""

# Create an instance of the P model and the Q model
nlpca_model = nlpca(hidden_space_dim, hidden_layer_dim, observed_space_dim)
nlpca_q = Q_nlpca(hidden_space_dim, hidden_layer_dim, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Define inference object
VI_nlpca = inf.inference.VI(nlpca_q, optimizer=optimizer, epochs=num_epochs)

# Fit the model
print("Fitting NLPCA model:")
nlpca_model.fit({"x": X}, VI_nlpca)

# Print loss function
print("\nNLPCA loss function evolution (image).")
print_loss_function(VI_nlpca.losses)

# Print posterior sample.
z = nlpca_model.posterior("z", data={"x": X}).sample()
print("NLPCA posterior sample (image).")
print_posterior(z, y)

# Test Separability
print("Separability results:")
test_separability(X, z, y)


###### Variational auto-encoder
"""
The data is generated supposing the hidden representation is generated using standard Gaussian distributions:
$$
   Z \sim \mathcal{N}_{hidden\_space\_dim}(0,I).
$$
A two layer network is used for the observed data's mean value:
$$
   X \mid z \sim \mathcal{N}_{observed\_space\_dim}(f(z), I).
$$

On the other hand, the variational distribution supposes the observed representation is generated using standard Gaussian distributions:
$$
   X \sim \mathcal{N}_{observed\_space\_dim}(0,I).
$$
A two layer network is used for the hidden data's mean and deviation value, more precisely the first components of the output are used for the mean and the last for the deviation. Let $[\mu, \sigma] = g(x)$, then
$$
   Z \mid x \sim \mathcal{N}_{observed\_space\_dim}(\mu, \sigma).
$$
"""

# create an instance of the P model and the Q model
vae_model = vae(hidden_space_dim, hidden_layer_dim, observed_space_dim)
q_vae = Q_vae(hidden_space_dim, hidden_layer_dim, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Create inference object
VI_vae = inf.inference.VI(q_vae, optimizer=optimizer, epochs=num_epochs)

# Fit model
print("Fitting VAE model:")
vae_model.fit({"x": X}, VI_vae)

# Print loss function
print("\nVAE loss function evolution (image).")
print_loss_function(VI_vae.losses)

# Print posterior sample
z = vae_model.posterior("z", data={"x": X}).sample()
print("VAE posterior sample (image).")
print_posterior(z, y)

# Test Separability
print("Separability results:")
test_separability(X, z, y)

##############################################
###### Reduction to 3-dimensional space ######
##############################################
print(" -> 3-Dimensional Results <- ")

# The same experiment is now tested in a three-dimensional reduction.
hidden_space_dim = 3

###### Probabilistic PCA
# create an instance of the P model and the Q model
pca_model = pca(hidden_space_dim, observed_space_dim)
pca_q = Q_pca(hidden_space_dim, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Define inference object
VI_pca = inf.inference.VI(pca_q, optimizer=optimizer, epochs=num_epochs)
# Training
print("Fitting PCA model.")
pca_model.fit({"x": X}, VI_pca)

# Print posterior sample.
z = pca_model.posterior("z", data={"x": X}).sample()
print("\nPCA posterior sample (image).")
print_posterior(z, y)

# Test separability
print("Separability results:")
test_separability(X, z, y)

###### Non-linear PCA
# create an instance of the P model and the Q model
nlpca_model = nlpca(hidden_space_dim, hidden_layer_dim, observed_space_dim)
nlpca_q = Q_nlpca(hidden_space_dim, hidden_layer_dim, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Define inference object
VI_nlpca = inf.inference.VI(nlpca_q, optimizer=optimizer, epochs=num_epochs)
# Training
print("Fitting NLPCA model.")
nlpca_model.fit({"x": X}, VI_nlpca)

# Print posterior sample
z = nlpca_model.posterior("z", data={"x": X}).sample()
print("\nNLPCA posterior sample (image).")
print_posterior(z, y)

print("Separability results:")
test_separability(X, z, y)

###### Variational auto-encoder

# create an instance of the P model and the Q model
vae_model = vae(hidden_space_dim, hidden_layer_dim, observed_space_dim)
q_vae = Q_vae(hidden_space_dim, hidden_layer_dim, observed_space_dim)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Define variational inference object
VI_vae = inf.inference.VI(q_vae, optimizer=optimizer, epochs=num_epochs)
# fit model
print("Fitting VAE model.")
vae_model.fit({"x": X}, VI_vae)

# Print posterior sample
z = vae_model.posterior("z", data={"x": X}).sample()
print("\nVAE posterior sample (image).")
print_posterior(z, y)

print("Separability results:")
test_separability(X, z, y)
