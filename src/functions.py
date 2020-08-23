#!/usr/bin/env python3

##########################################
################ IMPORTS #################
##########################################

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

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


def set_seed(SEED):
    tf.random.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


##########################################
############ Print functions #############
##########################################


def print_loss_function(VI):
    """
    Prints the ELBO evolution of a given variational inference model.
    Arguments:
    - VI: VI trained object from InferPy package: https://inferpy.readthedocs.io/projects/develop/en/latest/_modules/inferpy/inference/variational/vi.html
    """
    L = VI.losses
    plt.plot(range(len(L)), L)
    plt.xlabel("Epochs")
    plt.ylabel("-ELBO")
    plt.title("ELBO evolution")
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
