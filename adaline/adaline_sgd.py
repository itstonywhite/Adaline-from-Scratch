import numpy as np

# Stochastic Gradient Descent Adaline API
class AdalineSGD:
    """
    ADAptive LInear NEuron classifier.
    
    Parameters:
        eta : float
            Learning Rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
        shuffle : bool (default: True)
            Shuffles training data every epoch if True to prevent
            cycles.
        random_state : int
            Random number generator seed for random weight
            initialization.
            
    Attributes:
        w_ : 1d-array
            Weights after fitting.
        b_ : Scalar
            Bias unit after fitting.
        losses_ : list
            Mean squared error loss function value averaged over
            all training examples in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

# Tony White ✍️
