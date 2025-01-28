# Quadratic Convex Problem Solver
This folder contains Python files to generate, solve, and test instances of quadratic convex problems in the form 

    minimize    1/2 x^T Q x + x^T q
    subject to  sum_{i in I^k}x_i = 1, for k in K
                x >= 0

where K is a set of disjoint simplices, and I^k form a partition of the set {1,...,n} by using the active-set method. 

The structure of the code is modular, enabling clear separation of problem generation, algorithm implementation, and usage.
---
## Folder Structure

1. **`ASQPImplementation.ipynb`**
    - Python notebook containing the whole code with the experiments implementation included.

1. **`MatricesGenerator.py`** 
    - Generates the instances of a quadratic convex problem.  

2. **`HelperFunctions.py`**  
    - Contains utility functions and helper methods used by `ActiveSet.py`. 

3. **`ActiveSet.py`**  
    - Implements the steps of the active-set method.

4. **`SolveQP.py`**  
    - Solves a quadratic problem using the active-set method implemented in `ActiveSet.py`.  

5. **`FunctionsForExperiments.py`**  
    - Contains the functions to adapt the quadratic problems generated to the required format for the Clarabel and quadprog solvers.
    
6. **`Experiments.py`**
    - Demonstrates the usage of `SolveQP.py` with eight different cases for different choices of the matrix Q and the vector q.  

7. **Folder `SparseMatrices`** 
    - Contains the four sparse matrices generated with Matlab and exported in .mat format. The path of these two files should be updated in the `Experiments.py` and the `ASQPNotebookImplementation.ipynb` files.