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
import bayespy.plot as bpplt


def set_seed(SEED):
    """
    Set the same seed on NumPy, Random and TensorFlow.
    """
    tf.random.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def test_separability(X, z, y):
    """
    Trains an SVM from ScikitLearn with default parameters and shows resulted separability in both (X,y) and (z,y).
    Parameters:
    - X: Observed space dataset.
    - z: Hidden space dataset.
    - y: Labels.
    """
    svm = SVC()
    svm.fit(X, y)
    print("SVM score in observed space:", svm.score(X, y))
    svm.fit(z, y)
    print("SVM score in hidden space:", svm.score(z, y))


##########################################
############ Print functions #############
##########################################


def print_loss_function(
    losses, xlabel="Epochs", ylabel="-ELBO", title="ELBO evolution", save_path=None
):
    """
    Prints the evolution of the given array values. Given a variational inference object
    from InferPy, using its looses array might print its ELBO evolution.
    Arguments:
    - L: Array of loss function values
    - save_path: path to save the image.
    """
    plt.plot(range(len(losses)), losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # Save figure
    if save_path != None:
        plt.savefig(save_path)

    plt.show()


def print_class_pie_diagram(y, labels, save_path=None):
    """
    Print y's proportion in a pie diagram.
    Arguments:
    - y: Set of labels.
    - labels: Set of label names.
    - save_path: path to save the image.
    """
    prop_class = y.value_counts(normalize=True)
    figureObject, axesObject = plt.subplots()
    axesObject.pie(prop_class * 100, labels=labels, autopct="%1.2f", startangle=180)
    axesObject.axis("equal")
    plt.title("Class distribution")

    # Save figure
    if save_path != None:
        plt.savefig(save_path)

    plt.show()


def print_posterior(z, y, save_path=None):
    """
    Prints the model posterior in a 2D representation. The model is supposed to have the following variables:
    - z: Hidden 2-dimensoinal variable to represent.
    - x: Observed variable with dataset X.
    - save_path: path to save the image.
    """
    # Create a temporal dataframe to plot using sns.
    df = pd.DataFrame(data=z)
    df[y.name] = y
    fig = sns.pairplot(df, hue=y.name, palette="Set2", diag_kind="kde", height=2.5)

    if save_path != None:
        fig.savefig(save_path)


def plot_mixture_distplot(
    probabilities, labels, colors, component=0, n_components=1, save_path=None
):
    """
    Prints a density function showing the posterior probability of each class belongingto each component.
    Arguments:
    - probabilities: array-like containing each component probabilities.
    - labels: array-like with each class name.
    - colors: array-like with colors for each class.
    - component: indicates which component to use.
    - n_components: indicates the total ammount of components. If not 1, all of them are printed.
    - save_path: path to save the image.
    """

    # If more than 1 component needs to be plotted
    if n_components != 1:
        # Create subplot axes
        _, axes = plt.subplots(1, n_components, figsize=(20, 5), sharex=True)
        # For each component
        for index in range(n_components):
            # For each class
            for cl, (color, label) in enumerate(zip(colors, labels)):
                # Make the displot.
                sns.distplot(
                    probabilities[cl][:, index],
                    hist=False,
                    kde_kws={"color": color, "clip": (0.0, 1.0), "label": label},
                    ax=axes[index],
                )
    # When using only one component
    else:
        # for each class
        for (cl, color, label) in zip(classes, colors, labels):
            # Plot distplot
            sns.distplot(
                model.predict_proba(cl)[:, component],
                hist=False,
                kde_kws={"color": color, "clip": (0.0, 1.0), "label": label},
            )
    # Show legend
    plt.legend()

    # Save file
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
