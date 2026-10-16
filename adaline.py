# Importing dependencies & packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

            # Calculating loss after the update
            updated_output = self.activation(self.net_input(X))
            loss = ((y - updated_output) ** 2).mean()
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

# visualizing the decision boundaries for two-dimensional datasets
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plot the decision regions of a classifier for 2D data.

    Parameters:
        1. X : array-like, shape = [n_examples, 2]
            Feature matrix. Must have exactly two columns, since
            the plot is two-dimensional.
        2. y : array-like, shape = [n_examples]
            Class labels. At most 5 distinct classes.
        3. classifier : object
            Fitted model with a predict(X) method that accepts an
            array of shape [n_points, 2] and returns class labels.
        4. resolution : float, optional (default=0.02)
            Grid step size in feature units. Smaller values give a
            smoother boundary but need more predict calls.

    Returns:
        None
    """
    
    # Setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plotting the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class labels
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')

def main():
    # Training a Gradient Descent Adaline model on the Iris dataset
    
    # Loading the dataset
    df = pd.read_csv('./iris.txt', header=None, encoding='utf-8')

    # Select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    # Extract the sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # Standardization
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    # Training GD Adaline algorithm on the Iris data subset
    ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
    ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)

    # Training GD Adaline algorithm on standardized data
    ada_gd = AdalineGD(n_iter=20, eta=0.5)
    ada_gd.fit(X_std, y)

    # Loss vs. the number of epochs plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(range(1, len(ada1.losses_) + 1),
            np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - Learning rate 0.1')

    ax[1].plot(range(1, len(ada2.losses_) + 1),
            ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Mean squared error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()
    
    # Standardized Adaline's Decision Regions Plot
    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('Sepal Length [standardized]')
    plt.ylabel('Petal Length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # MSE by number of epochs plot
    plt.plot(range(1, len(ada_gd.losses_) + 1),
             ada_gd.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# Tony White ✍️
