# 🧠 Adaline Algorithm from Scratch

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-11557c.svg)](https://matplotlib.org/)

An implementation of the **ADAptive LInear NEuron (Adaline)** algorithm using Python and NumPy, built from scratch to explore the fundamentals of machine learning and gradient-based optimization.

The project uses an object-oriented design and the Iris dataset to demonstrate how a linear classifier learns from data, how the learning rate affects convergence, and how feature standardization influences gradient descent.

## 📚 About Adaline

**ADAptive LInear NEuron (Adaline)** is a linear classifier that learns by minimizing a cost function using gradient descent.

Unlike the Perceptron, which updates its weights based on classification errors, Adaline calculates errors using its continuous linear output.

### Linear Activation Function

For each input vector, Adaline calculates the net input and applies a linear (identity) activation function:

$$
z^{(i)} = \mathbf{w}^T\mathbf{x}^{(i)} + b
$$

$$
\sigma(z^{(i)}) = z^{(i)}
$$

Since the activation function is linear, its output is equal to the net input.

### Cost Function

Adaline minimizes the Mean Squared Error (MSE):

$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}\left(y^{(i)}-\sigma(z^{(i)})\right)^2
$$

### Parameter Updates

The weights and bias are updated in the direction that reduces the cost function, using a learning rate ($\eta$).

**Full Batch Gradient Descent (GD):**

$$
\mathbf{w} \leftarrow \mathbf{w} + \frac{2\eta}{n}\sum_{i=1}^{n}\left(y^{(i)}-\sigma(z^{(i)})\right)\mathbf{x}^{(i)}
$$

$$
b \leftarrow b + \frac{2\eta}{n}\sum_{i=1}^{n}\left(y^{(i)}-\sigma(z^{(i)})\right)
$$

Full Batch GD calculates the gradients using the entire training dataset and updates the parameters once per epoch.

**Stochastic Gradient Descent (SGD):**

$$
\mathbf{w} \leftarrow \mathbf{w} + 2\eta\left(y^{(i)}-\sigma(z^{(i)})\right)\mathbf{x}^{(i)}
$$

$$
b \leftarrow b + 2\eta\left(y^{(i)}-\sigma(z^{(i)})\right)
$$

SGD updates the weights and bias after each individual training sample rather than waiting to process the entire dataset.

Where:

- $n$ = Number of training examples.
- $y^{(i)}$ = True target value for the $i$-th sample.
- $z^{(i)}$ = Net input for the $i$-th sample.
- $\sigma(z^{(i)})$ = Continuous linear output for the $i$-th sample.
- $\mathbf{x}^{(i)}$ = Input feature vector.
- $\mathbf{w}$ = Weight vector.
- $b$ = Bias.
- $\eta$ = Learning rate.

## 📊 Results and Visualizations

### 1. Learning Rate Comparison (Full Batch GD)

Compare the training loss using different learning rates on the original, non-standardized Iris features.

![Adaline learning rate comparison (full batch gd)](./plots/loss%20vs%20epochs%20plot.png)

### 2. Decision Regions

Visualize the decision boundary learned by Adaline using standardized sepal length and petal length features.

![Adaline decision regions](./plots/Adaline%27s%20Decision%20Regions.png)

### 3. Mean Squared Error by Epoch

Observe how the model's Mean Squared Error changes as full batch gradient descent progresses on the standardized dataset.

![Adaline MSE by epoch](./plots/MSE%20by%20number%20of%20epochs%20plot.png)

### 4. Average Loss per Epoch in SGD

Observe the average loss recorded across individual sample updates during each epoch of stochastic gradient descent.

![Adaline Average loss by Epoch](./plots/Average%20loss%20plot.png)

## 🛠️ Implementation Details

The project provides two Adaline implementations:

| Class        | Description                               |
| ------------ | ----------------------------------------- |
| `AdalineGD`  | Trains using Full Batch Gradient Descent. |
| `AdalineSGD` | Trains using Stochastic Gradient Descent. |

Both implementations provide the following methods:

| Method          | Description                                            |
| --------------- | ------------------------------------------------------ |
| `fit(X, y)`     | Trains the model.                                      |
| `net_input(X)`  | Calculates the weighted sum plus bias.                 |
| `activation(X)` | Computes the linear activation.                        |
| `predict(X)`    | Predicts binary class labels using a threshold of 0.5. |

`AdalineSGD` also provides `partial_fit(X, y)` for incremental training without reinitializing the model parameters (Online Learning).

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

or

```bash
python examples/adaline_sgd_iris.py
```

The examples train Adaline models, display their loss curves, and visualize the learned decision regions.

## 📂 Repository Structure

```text
.
├── adaline/
│   ├── __init__.py
│   ├── adaline_gd.py        # Full Batch Gradient Descent
│   └── adaline_sgd.py       # Stochastic Gradient Descent
│
├── data/
│   └── iris.txt             # Iris dataset
│
├── examples/
│   ├── adaline_gd_iris.py   # Training & visualization example (GD)
│   └── adaline_sgd_iris.py  # Training & visualization example (SGD)
│
├── plots/
│   ├── Adaline's Decision Regions.png
│   ├── loss vs epochs plot.png
│   ├── MSE by number of epochs plot.png
│   └── Average loss plot.png
│
│
├── utils/
│   └── plotting.py          # Decision-region visualization utility
│
├── .gitignore
└── README.md                # Document
```

---

### 👨‍💻 Author

**Tony White**

GitHub: [@itstonywhite](https://github.com/itstonywhite)

_Built for learning, experimentation, and a deeper understanding of machine learning from first principles._
