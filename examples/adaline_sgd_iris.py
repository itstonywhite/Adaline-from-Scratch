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
from adaline import AdalineSGD
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


if __name__ == "__main__":
    main()
    
# Tony White ✍️
