
import numpy as np

# Stochastic Gradient Descent Adaline API
class AdalineSGD:
    """
    ADAptive LInear NEuron classifier.

    Parameters:
        eta : float
            Learning rate.
        n_iter : int
            Passes over the training dataset.
        shuffle : bool (default: True)
            Shuffles training data every epoch if True to prevent
            cycles.
        random_state : int or None
            Random number generator seed for random weight initialization.

    Attributes:
        w_ : 1d-array
            Weights after fitting.
        b_ : Scalar
            Bias unit after fitting.
        losses_ : list
            Average of the squared errors calculated for individual
            training examples during each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        """
        Fit training data.

        Parameters:
            X : {array-like}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the number of
                examples and n_features is the number of features.
            y : array-like, shape = [n_examples]
                Target values.

        Returns:
            self : object
        """
        # Input validation
        X, y = self._validate_data(X, y)

        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            losses = []

            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))

            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def partial_fit(self, X, y):
        """
        Fit training data without reinitializing the weights.

        Accepts either a single training example or a batch of examples.

        Parameters:
            X : array-like
                Shape = [n_features] for one example, or
                [n_examples, n_features] for a batch.
            y : array-like or scalar
                Target value(s).

        Returns:
            self : object
        """
        # Input validation
        X, y = self._validate_data(
            X, y, allow_single_sample=True
        )

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        elif X.shape[1] != self.w_.shape[0]:
            raise ValueError(
                "X must have the same number of features as the "
                "training data."
            )

        for xi, target in zip(X, y):
            self._update_weights(xi, target)

        return self

    def _validate_data(self, X, y, allow_single_sample=False):
        """Validate input data and convert it into NumPy arrays."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if allow_single_sample:
            if X.ndim == 1:
                X = X.reshape(1, -1)

            if y.ndim == 0:
                y = y.reshape(1)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if X.shape[0] == 0:
            raise ValueError("X cannot be empty.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")

        return X, y

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = self.rgn.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers."""
        self.rgn = np.random.default_rng(self.random_state)
        self.w_ = self.rgn.normal(
            loc=0.0, scale=0.01, size=m
        )
        self.b_ = np.float64(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights."""
        output = self.activation(self.net_input(xi))
        error = target - output

        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error

        loss = error ** 2
        return loss

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation."""
        return X

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(
            self.activation(self.net_input(X)) >= 0.5,
            1,
            0
        )


# Tony White ✍️
