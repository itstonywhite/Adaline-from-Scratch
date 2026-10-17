# 🧠 Adaline Algorithm from Scratch

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-11557c.svg)](https://matplotlib.org/)

An implementation of the **ADAptive LInear NEuron (Adaline)** algorithm using Python and NumPy, built from scratch to explore the fundamentals of machine learning and gradient-based optimization.

The project uses an object-oriented design and the Iris dataset to demonstrate how a linear classifier learns from data, how the learning rate affects convergence, and how feature standardization influences gradient descent.

## 📚 About Adaline

Adaline is a linear classifier that learns by minimizing a cost function using gradient descent.

Unlike the Perceptron, which updates its weights based on classification errors, Adaline uses the continuous output of its linear activation function to calculate the error during training.

### Cost Function

The model minimizes the Mean Squared Error (MSE):

$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}\left(y^{(i)}-\sigma(z^{(i)})\right)^2
$$

For each input vector, Adaline calculates a continuous linear output:

$$
\sigma(z^{(i)}) = \mathbf{w}^T\mathbf{x}^{(i)} + b
$$

Where:

- $n$ = Number of training examples
- $y^{(i)}$ = True target value for the $i$-th sample
- $\hat{y}^{(i)}$ = Continuous linear output for the $i$-th sample
- $\mathbf{x}^{(i)}$ = Input feature vector for the $i$-th sample
- $\mathbf{w}$ = Weight vector
- $b$ = Bias

The weights and bias are updated using full batch gradient descent.

## 📊 Results and Visualizations

### 1. Learning Rate Comparison

Compare the training loss using different learning rates on the original, non-standardized Iris features.

![Adaline learning rate comparison](./plots/loss%20vs%20epochs%20plot.png)

### 2. Decision Regions

Visualize the decision boundary learned by Adaline using standardized sepal length and petal length features.

![Adaline decision regions](./plots/Adaline%27s%20Decision%20Regions.png)

### 3. Mean Squared Error by Epoch

Observe how the model's Mean Squared Error changes as full batch gradient descent progresses on the standardized dataset.

![Adaline MSE by epoch](./plots/MSE%20by%20number%20of%20epochs%20plot.png)

## 🛠️ Implementation Details

The `AdalineGD` class provides the following methods:

| Method          | Description                                            |
| --------------- | ------------------------------------------------------ |
| `fit(X, y)`     | Trains the model using full batch gradient descent.    |
| `net_input(X)`  | Calculates the weighted sum plus bias.                 |
| `activation(X)` | Returns the linear activation.                         |
| `predict(X)`    | Predicts binary class labels using a threshold of 0.5. |

The implementation uses NumPy for vectorized mathematical operations and Matplotlib for visualization. Pandas is used to load and process the Iris dataset.

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/itstonywhite/Adaline-from-Scratch.git
cd Adaline-from-Scratch
```

2. Install the required dependencies:

```bash
pip install numpy pandas matplotlib
```

### Run the Example

Run the Iris experiment directly from the project root:

```bash
python examples/adaline_gd_iris.py
```

<!-- or

```bash
python examples/adaline_sgd_iris.py
``` -->

The example trains Adaline models using different learning rates, displays their loss curves, and visualizes the learned decision regions.

## 📂 Repository Structure

```text
.
├── adaline/
│   ├── __init__.py
│   ├── adaline_gd.py       # Full Batch Gradient Descent
│   └── adaline_sgd.py      # Stochastic Gradient Descent (in development)
│
├── data/
│   └── iris.txt            # Iris dataset
│
├── examples/
│   └── adaline_gd_iris.py  # Training and visualization example
│
├── plots/
│   ├── Adaline's Decision Regions.png
│   ├── loss vs epochs plot.png
│   └── MSE by number of epochs plot.png
│
├── utils/
│   └── plotting.py         # Decision-region visualization utility
│
├── .gitignore
└── README.md
```

---

### 👨‍💻 Author

**Tony White**

GitHub: [@itstonywhite](https://github.com/itstonywhite)

_Built for learning, experimentation, and a deeper understanding of machine learning from first principles._
