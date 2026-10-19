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
    pass


if __name__ == "__main__":
    main()
    
# Tony White ✍️
