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
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgn.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgn = np.random.default_rng(self.random_state)
        self.w_ = self.rgn.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * (error)
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""

# Tony White ✍️
