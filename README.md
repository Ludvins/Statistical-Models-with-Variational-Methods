# Statistical Models with Variational Methods


**Variational inference** concepts, which are adapted from statistical physics, first appeared in *A mean field theory learning algorithm for neural networks (James R Anderson and Carsten Peterson)*, in which the authors used them to fit a neural-network. More precisely, they used **mean-field** methods to achieve it.

In the coming years, several studies were done on variational inference, such as *Keeping the neural networks simple by minimizing the description length of the weights (Geoffrey E Hinton and Drew Van Camp)*, which used further mean-field methods in neural networks, and *An introduction to variational methods for graphical (Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola and Lawrence K Saul)* that generalized variational inference to many models.

Today, variational inference is more scalable and easy to derive, in some cases it is even automated. It has been applied to many different models and types of learning.

This document attempts to give an overview of some results in **Bayesian variational inference** as well as to test some frameworks for probabilistic modeling. In these tests an effort to apply variational inference techniques to real databases is made.

The theoretical part of this document, which is encompassed by chapters 1 to 23, describes the basic concepts of **statistical inference**, from classical to **variational**. After this, the **exponential family** and **graphical models** are reviewed together with their influence in variational inference, focusing on how the inference task is simplified by their usage.

On the other hand, the last chapters, focus on the utilization of different frameworks to experiment different models, which involve **Gaussian mixture** and dimensionality reduction via **principal components analysis** and **variational auto-encoders**.

The main sources used for writing this documents were 
- *Bayesian reasoning and machine learning (David Barber)*.
- *Pattern recognition and machine learning (Christopher M Bishop)*.
- *Probabilistic Graphical Models, Principles and Techniques (Daphne Koller and Nir Friedman)*.
- *Probabilistic Models with Deep Neural Networks (Andrés R. Masegosa, Rafael Cabañas, Helge Langseth, Thomas D. Nielsen and Antonio Salmerón)*.
- *Variational inference: A review for statisticians (David M Blei, Alp Kucukelbir and Jon D McAuliffe)*.
