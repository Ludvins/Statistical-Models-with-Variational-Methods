#!/usr/bin/env python3

##########################################
################ IMPORTS #################
##########################################

import math
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
import warnings
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import inferpy as inf
import sys
from inferpy.data import mnist

import matplotlib.cm as cm
from bayespy import nodes
from bayespy.inference import VB

from sklearn.mixture import GaussianMixture

##########################################
############ Print functions #############
##########################################


def print_loss_function(VI):
    """
    Prints the loss function evolution of a given variational inference model.
    Arguments:
    - VI: VI trained object from InferPy package: https://inferpy.readthedocs.io/projects/develop/en/latest/_modules/inferpy/inference/variational/vi.html
    """
    L = VI.losses
    plt.plot(range(len(L)), L)

    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.title("Loss evolution")
    plt.grid(True)
    plt.show()


def print_class_pie_diagram(y, labels):
    prop_class = y.value_counts(normalize=True)
    figureObject, axesObject = plt.subplots()
    axesObject.pie(prop_class * 100, labels=labels, autopct="%1.2f", startangle=180)
    axesObject.axis("equal")
    plt.title("Class distribution")
    plt.show()


def print_posterior(z, y):
    """
    Prints the model posterior in a 2D representation. The model is supposed to have the following variables:
    - z: Hidden 2-dimensoinal variable to represent.
    - x: Observed variable with dataset X.
    """
    df = pd.DataFrame(data=z)
    df[y.name] = y
    sns.pairplot(df, hue=y.name, palette="Set2", diag_kind="kde", height=2.5)


def print_posterior_with_line(z, y, w):
    """
    Prints the model posterior in a 2D representation. The model is supposed to have the following variables:
    - z: Hidden 2-dimensoinal variable to represent.
    - x: Observed variable with dataset X.
    """
    # Set markers and colors for each digit
    ax = sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=y, style=y, legend="full")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    v = np.array([xmin, xmax])
    ax.plot(
        v, (-w[0] - w[1] * v) / w[2],
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def test_separability(X, z, y):
    svm = SVC()
    svm.fit(X, y)
    print("SVM score in observed space:", svm.score(X, y))
    svm.fit(z, y)
    print("SVM score in hidden space:", svm.score(z, y))


##################################################
###################### PCA #######################
##################################################

# Parametric model
@inf.probmodel
def pca(k, d):
    w = inf.Normal(loc=tf.zeros([k, d]), scale=1, name="w")  # shape = [k,d]
    w0 = inf.Normal(loc=tf.zeros([d]), scale=1, name="w0")  # shape = [d]
    with inf.datamodel():
        z = inf.Normal(tf.zeros([k]), 1, name="z")  # shape = [N,k]
        x = inf.Normal(z @ w + w0, 1, name="x")  # shape = [N,d]


# Variational model
@inf.probmodel
def Q_pca(k, d):
    qw_loc = inf.Parameter(tf.zeros([k, d]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")

    qw0_loc = inf.Parameter(tf.ones([d]), name="qw0_loc")
    qw0_scale = tf.math.softplus(inf.Parameter(tf.ones([d]), name="qw0_scale"))
    qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")

    with inf.datamodel():
        qz_loc = inf.Parameter(np.zeros([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        qz = inf.Normal(qz_loc, qz_scale, name="z")


#################################################
##################### NLPCA #####################
##################################################


def decoder(k, d0, d1, loc_init=0.001, scale_init=1):
    """
    Decoder neural network for non-linear PCA.
    """
    beta0 = inf.Normal(tf.ones([k, d0]) * loc_init, scale_init, name="beta0")
    alpha0 = inf.Normal(tf.ones([d0]) * loc_init, scale_init, name="alpha0")

    ######

    beta1 = inf.Normal(tf.ones([d0, d1]) * loc_init, scale_init, name="beta1")
    alpha1 = inf.Normal(tf.ones([d1]) * loc_init, scale_init, name="alpha1")

    def call(z):
        h0 = tf.nn.relu(z @ beta0 + alpha0, name="h0")
        return h0 @ beta1 + alpha1

    return call


@inf.probmodel
def nlpca(k, d0, d1):
    decoder_call = decoder(k, d0, d1)
    with inf.datamodel():
        # Define local latent variables
        z = inf.Normal(loc=tf.ones([k]), scale=1, name="z")

        output = decoder_call(z)

        # Define the observed variables
        x = inf.Normal(loc=output, scale=1.0, name="x")


@inf.probmodel
def Q_nlpca(k, d0, d1, loc_init=0.001, scale_init=1):
    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        qz = inf.Normal(loc=qz_loc, scale=qz_scale, name="z")

    ###

    qbeta0_loc = inf.Parameter(tf.ones([k, d0]) * loc_init, name="qbeta0_loc")
    qbeta0_scale = tf.math.softplus(
        inf.Parameter(tf.ones([k, d0]) * scale_init, name="qbeta0_scale")
    )
    qbeta0 = inf.Normal(qbeta0_loc, qbeta0_scale, name="beta0")

    qalpha0_loc = inf.Parameter(tf.ones([d0]) * loc_init, name="qalpha0_loc")
    qalpha0_scale = tf.math.softplus(
        inf.Parameter(tf.ones([d0]) * scale_init, name="qalpha0_scale")
    )
    qalpha0 = inf.Normal(qalpha0_loc, qalpha0_scale, name="alpha0")

    ###

    qbeta1_loc = inf.Parameter(tf.ones([d0, d1]) * loc_init, name="qbeta1_loc")
    qbeta1_scale = tf.math.softplus(
        inf.Parameter(tf.ones([d0, d1]) * scale_init, name="qbeta1_scale")
    )
    qbeta1 = inf.Normal(qbeta1_loc, qbeta1_scale, name="beta1")

    qalpha1_loc = inf.Parameter(tf.ones([d1]) * loc_init, name="qalpha1_loc")
    qalpha1_scale = tf.math.softplus(
        inf.Parameter(tf.ones([d1]) * scale_init, name="qalpha1_scale")
    )
    qalpha1 = inf.Normal(qalpha1_loc, qalpha1_scale, name="alpha1")


##################################################
###################### VAE #######################
##################################################

# initial values
loc_init = 0.001
scale_init = 1
scale_epsilon = 0.01

scale_init_encoder = 0.01
scale_epsilon = 0.01


@inf.probmodel
def vae(k, d0, dx):
    with inf.datamodel():
        z = inf.Normal(tf.ones(k), 1, name="z")

        decoder = inf.layers.Sequential(
            [
                tf.keras.layers.Dense(d0, activation=tf.nn.relu),
                tf.keras.layers.Dense(dx),
            ]
        )

        x = inf.Normal(decoder(z), 1, name="x")


# Q model for making inference
@inf.probmodel
def Q_vae(k, d0, dx):
    with inf.datamodel():
        x = inf.Normal(tf.ones(dx), 1, name="x")

        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d0, activation=tf.nn.relu),
                tf.keras.layers.Dense(2 * k),
            ]
        )

        output = encoder(x)
        qz_loc = output[:, :k]
        qz_scale = tf.nn.softplus(output[:, k:]) + scale_epsilon
        qz = inf.Normal(qz_loc, qz_scale, name="z")
