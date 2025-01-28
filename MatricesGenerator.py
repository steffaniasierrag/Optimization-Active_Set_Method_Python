import numpy as np
import random
from scipy.sparse import csc_matrix

def generate_symmetric_psd_matrix(n,rank):
    assert rank <= n, "Rank k must be less than to n"

    # Random matrices for rank-k construction
    A = np.random.randn(n, rank)  # m x k matrix
    B = np.random.randn(rank, n)  # k x m matrix

    # Resultant matrix
    Q = np.dot(A, B)

    # Construct psd matrix by multiplying QQ^T
    Q = np.dot(Q,Q.T)
    return Q

   
def generate_symmetric_pd_matrix(n):
    '''Function that returns a symmetric positive definite matrix.'''
    
    def generate_full_rank_matrix(n):
        while True:
            G = np.random.rand(n, n)  # Generate random matrix of size n x n
            if np.linalg.matrix_rank(G) == n:  # Check if it's full rank
                return G  
            
    G = generate_full_rank_matrix(n)
    G_T = np.transpose(G)
    Q = np.dot(G_T, G)
    return Q

def define_partition(n, k):
    '''Function that returns a random partition of size k of the set {1...n}'''
    
    lst = list(range(0,n))
    random.shuffle(lst) # shuffle the list to compose sets with different elements
    indexes = sorted(random.sample(range(0,n-1), k - 1)) # up to n-1 bc we don't want to take as index the last one to don't generate empty subsets
    cand_partition = [np.array(lst[start:end]) for start, end in zip([0] + indexes, indexes + [len(lst)])]
    return cand_partition

def partition(n,k):
    while True:
        partition = define_partition(n,k)
        if all(len(subs) > 0 for subs in partition):
            return partition

def matrix_of_constraints(partition): 
    '''Function that returns the matrix of constraints (in sparse format) defined by the given partition.'''
    
    n = sum([len(subset) for subset in partition])
    partition = [subset for subset in partition]

    # extract the indexes of the positions with ones
    cols_idx = np.concatenate(partition)
    rows_idx = [idx for idx, set in enumerate(partition) for _ in range(len(set))]

    # create the matrix A
    A = np.zeros((len(partition), n))
    A[rows_idx, cols_idx] = 1

    A = csc_matrix(A)
    return A

def define_vector_b(k):
  'Generates a vector of ones of length k'
  b = np.ones((k,1))
  return b

# Generate the initial vector
def initial_point(partition):
    '''Generates the vector x0 of size n such that the i-th position is 1 
    if i is the smallest element in one of the subsets of the partition; 
    otherwise, it is 0.'''

    n = sum([len(subset) for subset in partition])
    positions_ones = [min(sub) for sub in partition]
    
    x0 = np.zeros((n,1))
    np.add.at(x0, positions_ones, 1) # increment items of x0 in position 'positions' by 1.
    return x0

def generate_c_vector(n):
    '''Generate the vector representing the linear part of the objective function'''
    c = np.random.rand(n)
    c = c.reshape((n,1))
    return c

def generate_special_c_vector(Q,x_0):
    c = np.matmul(Q,x_0)*-(1)
    return c