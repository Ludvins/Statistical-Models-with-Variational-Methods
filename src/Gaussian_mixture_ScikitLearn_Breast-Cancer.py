#!/usr/bin/env python
# coding: utf-8

# The aim of this script is to test Scikit-Learn potential on modelling Gaussian mixtures.
# Two attepts are done, the first on the proper Breast Cancer Wisconsin dataset, and the
# second on a three-dimensional reduction using a variational auto-encoder.

# The mixture is componed by the same number of components as classes has the dataset.
# An optimal outcome would be that each class is modeled by one component.
# In order to test this, we are using each datapoint posterior probability to belong
# to each component to compute a density function over each class and component.


# Imports
from models import *
from functions import *
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split

# seed
np.random.seed(2020)

# Load and prepare Breast Cancer database, located in ```../database/```.
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

# We define a function that separates and organizes the dataset in classes.
def separate_in_classes(X, y):
    return (X[y == "B"], X[y == "M"])


# Ammount of samples
n_samples = X.shape[0]
# Observed space dimensionality
data_dim = X.shape[1]
# Ammount of classes
n_classes = len(y.unique())
# Hidden representation dimension
h_dim = 3


def gm_model(n_classes, dim):
    """
    Create `BayesianGaussianMixture` object from [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html), with a number of components equal to the ammount of classes.

    The parameter `weight_concentration_prior_type` models if a Dirichlet distribution
    or a Dirichlet Process is used for the weights. In this case, we model using a distribution.

    We are setting covariance_type = "diag" to reduce the ammount of parameters that need to
    be learned form the model.

    The following prior values are used:
    - The means variable $\mu$ follows a centered normal distribution:
    $$\mu_k \sim \mathcal{N}_{data\_dim}(0,I).$$
    - The precision variables $\Lambda$ follow a Wishart distribution with parameters
    $$\Lambda_k \sim \mathcal{W}_{data\_dim}(data\_dim, I).$$
    The covariance matrix are restricted to diagonal matrices.
    - The weights concentration variable follows a dirichlet with parameter
    $$ \pi \sim \text{Symmetric-Dirichlet}\Big(\frac{1}{n\_classes}\Big).$$
    """
    return BayesianGaussianMixture(
        n_components=n_classes,
        weight_concentration_prior_type="dirichlet_distribution",
        weight_concentration_prior=1 / n_classes,
        covariance_type="diag",
        mean_precision_prior=1,
        mean_prior=np.zeros([dim]),
        degrees_of_freedom_prior=dim,
        covariance_prior=np.ones(dim),
        max_iter=1000,
        tol=1e-3,
        random_state=0,
    )


# We fit the model using the observed data `X`.
# The stop criteria is either reaching `max_iter` steps on the EM algorithm
# or having a lower bound difference betweeen iterations below `tol`.
gm = gm_model(n_classes, data_dim)
gm.fit(X)

# The model's posterior parameters might be inspected via `weights_`, `means_` and `precisions_`
# The mixture weights are:
print("Learned model weights: ")
print(gm.weights_)
input("Press Enter to continue...")

# Each component mean is:
print("Learned model means: ")
print(gm.means_)
input("Press Enter to continue...")

# Each component diagonal covariance matrix is:
print("Learned model covariance diagonal values: ")
print(gm.covariances_)
input("Press Enter to continue...")

# The dataset is split between the two classes
(B, M) = separate_in_classes(X, y)

# `predict_proba(p)` gives the probability of`p` belonging to each component.
# In particular, this shows that the benign case, do mostly belong to the second component.
print("Each component probability for Benign cases:")
print(gm.predict_proba(B))

# This might be shown using `plot_mixture_distplot`, which plots each component belonging density.
print("Component belonging densities (image).")
plot_mixture_distplot(
    [gm.predict_proba(B), gm.predict_proba(M)],
    ["Benign", "Malign"],
    ["blue", "green"],
    n_components=n_classes,
)

# The same procedure is now done under a lower dimensional representation of the data. A Variational auto-encoder is used.
vae_model = vae(h_dim, 100, data_dim)
q_vae = Q_vae(h_dim, 100, data_dim)

optimizer = tf.train.AdamOptimizer(0.01)
VI_vae = inf.inference.VI(q_vae, optimizer=optimizer, epochs=4000)
print("Variational auto-autoencoder training:")
vae_model.fit({"x": X}, VI_vae)

# Draw a posterior sample as datapoints.
data = vae_model.posterior("z", data={"x": X}).sample()

# We are now able to plot the data we are trying to fit into two Gaussian distributions.
print("\nModel posterior sample (image).")
print_posterior(data, y)

# Define and train the mixture model.
gm = gm_model(n_classes, h_dim)
gm.fit(data)

# The learned mean values can be shown.
# One may notice that the first component seeks to model the
# malign points and the second the benign ones.
print("Learned means:")
print(gm.means_)


# The same component distribution density is ploted for this results.
(B, M) = separate_in_classes(data, y)

print("Component belonging densities (image)")
plot_mixture_distplot(
    [gm.predict_proba(B), gm.predict_proba(M)],
    ["Benign", "Malign"],
    ["blue", "green"],
    n_components=n_classes,
)
