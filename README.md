# AMGNNv1.0

# AMGNN – Amine Guettara Neural Network

Hi! This is AMGNN, my custom neural network framework built entirely in C++.

Features

     Optimization Methods:

        Gradient Descent

        Mini-Batch Gradient Descent

        Stochastic Gradient Descent

     Optimizers:

        AMGNNO – Supports multiple learning rate decay functions (step, exponential, etc.)

        ADAM

     Differentiation:

        Uses interpolational polynomial methods to calculate gradients

        Supports three differentiation orders: 2, 4, and 6

     Task Types Supported:

        Regression

        Multiclass Classification

        Binary Classification

        Multilabel Classification

     Multithreaded Training:

        Threading for weights

        Threading for batches

        Threading for stochastic samples

        Thread usage is fully configurable

        Efficient Thread Pooling to adapt to your CPU

     Model Persistence:

        Save and load your models easily — portable and ready for reuse

 Limitations (for now)

    No GPU acceleration yet — planned for the future once GPU access is available (CUDA/OpenCL support coming soon)

 Coming Soon

     L-BFGS Optimizer (Quasi-Newton method)

     Full Documentation covering the framework’s internal structure, setup, and examples
