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
        
    def fit(self, X, y):
        """
        Fit training data.
        
        Parameters:
            X : {array-like}, shape = [n__examples, n_features]
                Training vectors, where n_examples is the number of
                examples and n_features is the number of features.
            y : array-like, shape = [n_examples]
                Target values.
                
        Returns:
            self : object.
        """
        
        self._initialize_weights(X.shape)
        self.losses_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.revel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
            
        return self
        

# Tony White ✍️
