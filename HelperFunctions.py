import numpy as np
from itertools import groupby
from scipy.sparse import csr_matrix, csc_matrix, eye, vstack # types of sparse matrices, and to stack vertic. matrices
from scipy.linalg import cho_factor, cho_solve # cholesky factorization

def find_the_partition(A):
    '''Returns the partition associataed to the given matrix A of constraints'''
    
    positions = np.argwhere(A==1)
    
    #to use groupby the iterable needs to be sorted on the key function. np.argwhere returns it sorted
    partition = [[col for row, col in group] for idxset, group in groupby(positions, key=lambda x: x[0])]
    
    return partition

def find_permutation_matrix(U, partition):
    """ Find the permutation matrix P given a partition and a set U of column indexes such that 
        AP = [I|Az], where A is a matrix of constraints"""    
    els_U= [list(set(Pi).intersection(U)) for Pi in partition]

    cumul_len = 0
    cols_idxs = []
    k = len(els_U)

    for i, subs in enumerate(els_U):
        
        cols_idxs.append(i)
        n_dependent_cols = len(subs)-1
        dep_pos = [k+cumul_len+j for j in range(n_dependent_cols)]
        cols_idxs.extend(dep_pos)
        cumul_len = cumul_len + n_dependent_cols

    U_flatten = [idx for subs in els_U for idx in subs]
    U_flatten_copy = U_flatten.copy()
    U_flatten.sort()
    
    new_positions = [(el_u, i) for i, el_u in enumerate(U_flatten)]
    ordered_new_positions = sorted(new_positions, key=lambda x: U_flatten_copy.index(x[0]))
    
    rows_idxs = [x[1] for x in ordered_new_positions]

    data = np.ones(len(rows_idxs))

    permutation_matrix = csc_matrix((data, (rows_idxs, cols_idxs)), shape=(len(rows_idxs),len(rows_idxs)))

    return permutation_matrix

def compute_Z(AuPerm, k, u):
    ''' Returns the matrix Z = [-A_z] used to find the direction p
                               [  I ] '''  
    AZ = AuPerm[:, k:]
    id_Z = eye(u-k)
    Z = vstack((-AZ, id_Z))
    return Z

def compute_pz(Z, perm_m, Qu, Y, py, g_u):
    '''pz is the vector obtained by solving the equation system
        (PZ)^TQ_U(PZ)pz = -(PZ)^T(QuPYpy+gu)
        where P=perm_m'''
    
    if Z.size == 0:
        pz = np.empty(0)
        return 1, pz
    else:
        # Find (PZ)^TQ_u(PZ),
        PZ = perm_m@Z
        PZTQ = PZ.transpose()@Qu
        PZQPZ = PZTQ@PZ.toarray()
        
        try:
            # Z^TQ_UYpy
            PY = perm_m@Y
            QPY = Qu@PY
            QPYPy = QPY@py

            term_r = -QPYPy - g_u
            term_r = PZ.transpose()@term_r
            term_r = term_r

            # Find the Cholesky decomposition of (PZ)^TQPZ.
            # If the matrix is singular we get an error.
            ch, low = cho_factor(PZQPZ) 

            # Solve the linear equation (PZ)^TQPZ pz = term_r
            pz = cho_solve((ch, low), term_r)
            return 1, pz
        except Exception as e:
            # print(f"Cholesky factorization failed: {e}")
            return 0, []

def compute_lambdas(perm_m, Y, g_u, Qu, p_u):
    '''Compute the Lagrange multipliers of the equality constraints as (PY)^T(gu+Qupu)'''
    PY = perm_m@Y
    Qup_u = Qu@p_u
    r_term = g_u+Qup_u

    lambdas = PY.T@r_term
    return lambdas

def compute_mult_ineq(Q, x, c, mult_eq, A):
    '''Compute the Lagrange multipliers of the inequality constraints as Qx+c-A^T*lambda'''
    Qx = np.matmul(Q, x)
    ATLambda = A.T@mult_eq
    mu = Qx+c-ATLambda
    return mu

def matrix_vector_mult(matrix_, vector_):
    '''Reshape the empty matrix obtained from the multiplication of either an empty matrix or an empty vector'''
    mat_vec = matrix_@vector_
    if matrix_.size == 0 or vector_.size==0:
        mat_vec = mat_vec.reshape(matrix_.shape[0],1)
    return mat_vec