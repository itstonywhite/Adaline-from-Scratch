# Importing dependencies & packages
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the project root discoverable by Python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing modules
from adaline import AdalineGD
from utils.plotting import plot_decision_regions


def main():

    # Locate project root and load dataset
    data_path = PROJECT_ROOT / "data" / "iris.txt"

    df = pd.read_csv(
        data_path,
        header=None,
        encoding="utf-8"
    )

    # Select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 0, 1)

    # Extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # Standardization
    X_std = np.copy(X)

    X_std[:, 0] = (
        X[:, 0] - X[:, 0].mean()
    ) / X[:, 0].std()

    X_std[:, 1] = (
        X[:, 1] - X[:, 1].mean()
    ) / X[:, 1].std()

    # Train Adaline models on original data
    ada1 = AdalineGD(
        n_iter=15,
        eta=0.1
    ).fit(X, y)

    ada2 = AdalineGD(
        n_iter=15,
        eta=0.0001
    ).fit(X, y)

    # Train Adaline on standardized data
    ada_gd = AdalineGD(
        n_iter=20,
        eta=0.5
    )

    ada_gd.fit(X_std, y)

    # Loss vs. number of epochs
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4)
    )

    ax[0].plot(
        range(1, len(ada1.losses_) + 1),
        np.log10(ada1.losses_),
        marker="o"
    )

    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log(Mean squared error)")
    ax[0].set_title("Adaline - Learning rate 0.1")

    ax[1].plot(
        range(1, len(ada2.losses_) + 1),
        ada2.losses_,
        marker="o"
    )

    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Mean squared error")
    ax[1].set_title("Adaline - Learning rate 0.0001")

    plt.tight_layout()
    plt.show()

    # Decision regions
    plot_decision_regions(
        X_std,
        y,
        classifier=ada_gd
    )

    plt.title("Adaline - Gradient Descent")
    plt.xlabel("Sepal Length [standardized]")
    plt.ylabel("Petal Length [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # MSE vs. number of epochs
    plt.plot(
        range(1, len(ada_gd.losses_) + 1),
        ada_gd.losses_,
        marker="o"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    
# Tony White ✍️
