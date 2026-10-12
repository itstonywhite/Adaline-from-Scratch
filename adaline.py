import numpy as np

class AdalineGD:
    """
    ADAptive LInear NEuron classifier.
    
    Parameters:
        eta = float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
        random_state : int
            Random number generator seed for random weight initialization.
            
    Attributes:
        w_ : 1d-array
            Weights after fitting.
        b_ : Scaler
            Bias unit after fitting.
        losses_ : list
            Mean squared error loss function values in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

