# L0-Penalized Lambda Search

This repository contains the implementation of a Branch-and-Bound (BnB) algorithm designed to solve least squares problems with an L0-norm penalty, promoting sparsity in the solution vector. This work is part of a research project conducted in France, aiming to explore the impact of penalization on sparse least-squares solutions.

## Overview

The primary objective of this project is to develop an algorithm that identifies intervals of the penalization parameter (λ) where the optimal solution remains unchanged. By leveraging the Fenchel dual formulation of the relaxed problem, we establish conditions for infeasibility, thereby delineating the parameter range where the solution is valid. The methodology is detailed in our report, "Impact of Penalization on Sparse Least-Squares Solutions."

## Repository Structure

- `BnB_normal.py`: Contains the main implementation of the Branch-and-Bound algorithm.
- `LSA.py`: Implements the Least Squares Approximation with L0 penalization.
- `node.py`: Defines the data structure for nodes used in the BnB algorithm.
- `utils.py`: Utility functions supporting the main algorithms.
- `Testing.ipynb`: Jupyter Notebook demonstrating the usage and testing of the implemented algorithms.
- `exp.jl`: Julia script used for experimental comparisons.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `CITATION.cff`: Citation file for referencing this work.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed and jupyter.


## Running the Branch-and-Bound Algorithm

To execute the BnB algorithm for the least squares problem with $L_0$ penalization, run:

```bash
python BnB_normal.py
```

This will initiate the algorithm with default parameters. For custom configurations, you can modify the parameters within the script or extend the code to accept command-line arguments

## Testing and Examples

The ``Testing.ipynb`` notebook provides examples and tests demonstrating the functionality of the implemented algorithms. It includes:

- Problem setup and parameter initialization.
- Execution of the BnB algorithm.
- Analysis of results and visualization.

To view and run the notebook:

- Ensure you have Jupyter Notebook installed.
- Navigate to the repository directory.
- Launch the notebook:

```bash
jupyter notebook Testing.ipynb
```

## Methodology

The Branch-and-Bound algorithm is employed to handle the NP-hard nature of the $L_0$-penalized least squares problem. Each node in the search tree represents a decision point for the inclusion or exclusion of variables in the solution vector. The algorithm alternates between processing the current node (bounding step) and selecting the next node to process (branching step). Detailed explanations of these steps are provided in our report.

## Results and Validation

Our approach was tested on simplified cases, such as using an identity matrix as the dictionary and a given vector ``y``. The results indicated that the identified interval encompassed the entire parameter space, providing no additional insights. While the method shows promise, further research and alternative implementations are necessary to obtain more informative results.

## Acknowledgments

We extend our gratitude to all individuals who provided guidance and supervision for this project at CentraleSupélec, Rennes.

## References

For a comprehensive understanding of the methodology and results, please refer to our report:

Pasche, M., Di Fatta, G., Dieudonne, G., Montant, T., Delecourt, C., Huhardeaux, L., & Elvira, C. (2025). Impact of Penalization on Sparse Least-Squares Solutions. CentraleSupélec, Rennes, France.
