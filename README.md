# AMGNNv1.0

AMGNN – Amine Guettara Neural Network

Hi! This is AMGNN, my custom neural network framework built entirely in C++.

Features

     Supports multiple optimization methods:

        Gradient Descent

        Mini-Batch Gradient Descent

        Stochastic Gradient Descent

     Two optimizers available:

        AMGNNO (supports various learning rate decay functions)

        ADAM

     Handles 4 task types:

        Regression

        Multiclass Classification

        Binary Classification

        Multilabel Classification

     Multithreaded Optimization:

        Threading for weights

        Threading for batches

        Threading for stochastic samples

        Configurable thread count

        Uses a Thread Pooling system to fit your CPU capabilities

     Model Saving & Loading:

        Easily save and load models — portable and ready for deployment

 Limitations (for now)

    No GPU acceleration (yet!). I don’t have a GPU at the moment — but once I get one, I plan to add CUDA/OpenCL support.

 Coming Soon

     L-BFGS Optimizer (Quasi-Newton method)

     Full documentation explaining the architecture, usage, and customization
