import numpy as np

# Full Batch Gradient Descent Adaline API
class AdalineGD:
    """
    ADAptive LInear NEuron classifier.
    
    Parameters:
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
        random_state : int
            Random number generator seed for random weight initialization.
            
    Attributes:
        w_ : 1d-array
            Weights after fitting.
        b_ : Scalar
            Bias unit after fitting.
        losses_ : list
            Mean squared error loss function values in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit training data.
        
        Parameters:
            X : {array-like}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the
                number of examples and n_features is the
                number of features.
            y : array-like, shape = [n_examples]
                Target values.
                
        Returns
            self : object
        """
        # Input validation
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if X.shape[0] == 0:
            raise ValueError("X cannot be empty.")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")

        # Converting inputs into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        rgn = np.random.default_rng(self.random_state)
        self.w_ = rgn.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            # Updating weights and bias unit
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()

            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))
                        >= 0.5, 1, 0)

# Tony White ✍️
