#!/usr/bin/env python3

import tensorflow as tf
from tensorflow_probability import edward2 as ed
import inferpy as inf
import numpy as np

##################################################
###################### PCA #######################
##################################################

# Parametric model
@inf.probmodel
def pca(k, d):
    """
    Probabilistic PCA model.
    Arguments:
    - k: hidden space dimension.
    - d: observed space dimension.

    The observed data is supossed to be generated as N(w^T z + delta, 1)
    """
    # Variable which encloses the linear transformation between spaces.
    # W ~ N(0,1)
    w = inf.Normal(loc=tf.zeros([k, d]), scale=1, name="w")

    # Variable that enables the observed data to be non-centered.
    # delta ~ N(0,1)
    delta = inf.Normal(loc=tf.zeros([d]), scale=1, name="delta")

    with inf.datamodel():
        # Variable that handles the hidden space representation of the data
        # Z ~ N(0,1)
        z = inf.Normal(tf.zeros([k]), 1, name="z")
        # Observed variables
        # X ~ N(w^T z + delta, 1)
        x = inf.Normal(z @ w + delta, 1, name="x")


# Variational model
@inf.probmodel
def Q_pca(k, d):
    """
    Variational model for Probabilistic PCA model (pca(k,d) function).
    Arguments:
    - k: hidden space dimension.
    - d: observed space dimension.

    """
    # W's mean parameter
    qw_loc = inf.Parameter(tf.zeros([k, d]), name="qw_loc")
    # W's deviation parameter
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d]), name="qw_scale"))
    # W ~ N(qw_loc, qw_scale)
    qw = inf.Normal(qw_loc, qw_scale, name="w")

    # delta's mean parameter
    qd_loc = inf.Parameter(tf.ones([d]), name="qd_loc")
    # delta's deviation parameter
    qd_scale = tf.math.softplus(inf.Parameter(tf.ones([d]), name="qd_scale"))
    # delta ~ N(qd_loc, qd_scale)
    qd = inf.Normal(qd_loc, qd_scale, name="delta")

    with inf.datamodel():
        # Z's mean parameter
        qz_loc = inf.Parameter(np.zeros([k]), name="qz_loc")
        # Z's deviation parameter
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        # Z ~ N(qz_loc, qz_scale)
        qz = inf.Normal(qz_loc, qz_scale, name="z")


#################################################
##################### NLPCA #####################
##################################################


def decoder(k, l, d):
    """
    Decoder two-layer neural network for non-linear PCA.
    It is composed by four N(0,1) distributions and a ReLU function.
    Arguments:
    - k: hidden space dimension.
    - l: neural network hidden layer dimension.
    - d: observed space dimension.

    No activation function is needed at the end as the value as the mean of a distribution.
    """

    # first layer
    beta0 = inf.Normal(tf.zeros([k, l]), 1, name="beta0")
    alpha0 = inf.Normal(tf.zeros([l]), 1, name="alpha0")

    # second layer
    beta1 = inf.Normal(tf.zeros([l, d]), 1, name="beta1")
    alpha1 = inf.Normal(tf.zeros([d]), 1, name="alpha1")

    # Define network computation. The decoder returns this function so that the four variables
    # are defined only one time.
    def operation(z):
        return tf.nn.relu(z @ beta0 + alpha0) @ beta1 + alpha1

    return operation


@inf.probmodel
def nlpca(k, l, d):
    """
    Probabilistic non-linear PCA model.
    Arguments:
    - k: hidden space dimension.
    - l: neural network hidden layer dimension.
    - d: observed space dimension.

    The observed data is supossed to be generated as X ~ N(f(z), 1).
    Where f is a two-layer neural network (k-l-d) in function decoder.
    """

    # Define the NN object
    nn = decoder(k, l, d)
    with inf.datamodel():
        # Variable that handles the hidden space representation of the data
        # Z ~ N(0,1)
        z = inf.Normal(loc=tf.zeros([k]), scale=1, name="z")

        # Retrieve the NN output
        output = nn(z)

        # Observed variables
        # X ~ N(nn(z), 1)
        x = inf.Normal(loc=output, scale=1.0, name="x")


@inf.probmodel
def Q_nlpca(k, l, d):
    """
    Variational model for Probabilistic non-linear PCA model (nlpca(k,l,d) function).
    Arguments:
    - k: hidden space dimension.
    - l: neural network hidden layer dimension.
    - d: observed space dimension.

    """

    # First layer

    # beta0's mean parameter
    qbeta0_loc = inf.Parameter(tf.zeros([k, l]), name="qbeta0_loc")
    # beta0's deviation parameter
    qbeta0_scale = tf.math.softplus(inf.Parameter(tf.ones([k, l]), name="qbeta0_scale"))
    # beta0 ~ N(qbeta0_loc, qbeta0_scale)
    qbeta0 = inf.Normal(qbeta0_loc, qbeta0_scale, name="beta0")

    # alpha0's mean parameter
    qalpha0_loc = inf.Parameter(tf.zeros([l]), name="qalpha0_loc")
    # alpha0's deviation parameter
    qalpha0_scale = tf.math.softplus(inf.Parameter(tf.ones([l]), name="qalpha0_scale"))
    # alpha0 ~ N(qalpha0_loc , qalpha0_scale)
    qalpha0 = inf.Normal(qalpha0_loc, qalpha0_scale, name="alpha0")

    # Second layer

    # beta1's mean parameter
    qbeta1_loc = inf.Parameter(tf.zeros([l, d]), name="qbeta1_loc")
    # beta1's deviation parameter
    qbeta1_scale = tf.math.softplus(inf.Parameter(tf.ones([l, d]), name="qbeta1_scale"))
    # beta1 ~ N(qbeta1_loc, qbeta1_scale)
    qbeta1 = inf.Normal(qbeta1_loc, qbeta1_scale, name="beta1")

    # alpha1's mean parameter
    qalpha1_loc = inf.Parameter(tf.zeros([d]), name="qalpha1_loc")
    # alpha1's deviation parameter
    qalpha1_scale = tf.math.softplus(inf.Parameter(tf.ones([d]), name="qalpha1_scale"))
    # alpha1 ~ N(qalpha1_loc , qalpha1_scale)
    qalpha1 = inf.Normal(qalpha1_loc, qalpha1_scale, name="alpha1")

    with inf.datamodel():
        # z's mean parameter
        qz_loc = inf.Parameter(tf.zeros([k]), name="qz_loc")
        # z's deviation parameter
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        # z ~ N(qz_loc, qz_scale)
        qz = inf.Normal(loc=qz_loc, scale=qz_scale, name="z")


##################################################
###################### VAE #######################
##################################################


@inf.probmodel
def vae(k, l, d):
    """
    Variational auto-encoder model.
    Arguments:
    - k: hidden space dimension.
    - l: neural network hidden layer dimension.
    - d: observed space dimension.

    The observed data is supossed to be generated as X ~ N(f(z), 1).
    Where f is a two-layer neural network (k-l-d) in function decoder.
    """

    # Network definition with Keras
    nn = inf.layers.Sequential(
        [tf.keras.layers.Dense(l, activation=tf.nn.relu), tf.keras.layers.Dense(d),]
    )

    with inf.datamodel():
        # Hidden variable representation Z ~ N(0,1)
        z = inf.Normal(tf.zeros(k), 1, name="z")
        # Observed variable X ~ N(nn(z), 1)
        x = inf.Normal(nn(z), 1, name="x")


# Q model for making inference
@inf.probmodel
def Q_vae(k, l, d):
    """
    Variational auto-encoder variational model.
    Arguments:
    - k: hidden space dimension.
    - l: neural network hidden layer dimension.
    - d: observed space dimension.

    The hidden data is supossed to be generated as Z ~ N(f(x)[:k], f(x)[k:]).
    Where f is a two-layer neural network (d-l-2k) in function decoder.
    """

    # Neural network definition
    nn = tf.keras.Sequential(
        [tf.keras.layers.Dense(l, activation=tf.nn.relu), tf.keras.layers.Dense(2 * k),]
    )

    with inf.datamodel():
        # Obsserved variable X ~ N(0,1)
        x = inf.Normal(tf.zeros(d), 1, name="x")

        # Network output
        output = nn(x)
        # The first k-terms correspond to the distribution's mean.
        qz_loc = output[:, :k]
        # The last k-terms correspond to the distribution's deviation. Using a softplus function
        # to avoid negative and 0 values. An offset is used to avoid getting 0 due to aproximation issues.
        qz_scale = tf.nn.softplus(output[:, k:]) + 0.001

        qz = inf.Normal(qz_loc, qz_scale, name="z")


##################################################
############### GAUSSIAN MIXTURE #################
##################################################


@inf.probmodel
def mixture(k, d):
    """
    Gaussian mixture model.
    Arguments:
    - k: number of components.
    - d: observed space dimensionality.
    """

    # Pi models the categorical parameter ruling each component probability.
    pi = inf.Dirichlet(
        np.ones(k) / k, allow_nan_stats=False, validate_args=True, name="pi"
    )

    # Lambda models each component precision using an inverse wishart distribution (inverse gamma multidimensional).
    Lambda = inf.InverseGamma(
        concentration=tf.ones([d, k]),
        scale=1,
        allow_nan_stats=False,
        validate_args=True,
        name="Lambda",
    )

    # Mu models each component mean value, using a Gaussian distribution.
    mu = inf.Normal(
        loc=tf.zeros([d, k]),
        scale=1,
        allow_nan_stats=False,
        validate_args=True,
        name="mu",
    )

    # As categorical distributions cannot be used, MixtureGaussian to model both the observed data and the categorical variable.
    with inf.datamodel():
        x = inf.MixtureGaussian(
            locs=mu,
            scales=Lambda,
            probs=pi,
            allow_nan_stats=False,
            validate_args=True,
            name="x",
        )


@inf.probmodel
def Q_mixture(k, d):
    """
    Gaussian mixture variational model.
    Arguments:
    - k: number of components.
    - d: observed space dimensionality.

    """
    # Dirichlet distribution for each component probability.
    qpi_param = inf.Parameter(tf.ones(k) / k, name="qpi_param")
    qpi = inf.Dirichlet(qpi_param, allow_nan_stats=False, validate_args=True, name="pi")

    # InverseGamma parameters and distribution.
    qLambda_w = inf.Parameter(tf.ones([d, k]), name="qLambda_w")
    qLambda_v = inf.Parameter(tf.ones([d, k]), name="qLambda_v")
    qLambda = inf.InverseGamma(
        concentration=tf.math.softplus(qLambda_w) + 0.01,
        scale=tf.math.softplus(qLambda_v) + 0.01,
        validate_args=True,
        allow_nan_stats=False,
        name="Lambda",
    )

    # Gaussian parameters and distribution.
    qmu_m = inf.Parameter(tf.zeros([d, k]), name="qmu_m")
    qmu_b = tf.math.softplus(inf.Parameter(tf.ones([d, k]), name="qmu_b"))
    qmu = inf.Normal(qmu_m, qmu_b, allow_nan_stats=False, validate_args=True, name="mu")
