import HelperFunctions as intermfns
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import null_space

def activesetmethod2(Q, c, A, b, x, L, U, n, k, partition, iter, verbose):
    """
    Solve a convex quadratic program:
        minimize    1/2 x^T Q x + x^T q
        subject to  sum_{i in I^k}x_i = 1, for k in K
                    x >= 0
        where K is a set of disjoint simplices, and I^k form a partition of the set {1,...,n}.

        Args:
            Q: nxn symmetric positive semidefinite matrix, the matrix Q from above.
            c: nx1 vector representing the linear part of the quadratic problem.
            A: kxn matrix representing the equality constraints.
            b: kx1 constraint vector from above.
            x: nx1 vector representing an approximated solution
            L: list with the indices of active variables
            U: list with the elements of {1,...,n}\L
            partition: list with the partition defined by A.
            iter: current iteration number
            verbose: True if the user want to print at each iteration the current objective value and the modifications to the active set
        
        Returns:
            x: nx1 vector representing an approximated solution of the quadratic programming problem.
            L: list of indices of the active variables.
            U: list of indices of the non-active variables.
            res: a string describing the status of the end of the iteration, with the following possible values:
                - 'the solution is optimal'
                - 'the solution is not optimal'
  """
    def log(message):
        if verbose:
            print(message)

    log('It {}'.format(iter))

    U.sort() 
    u = len(U)

    # Problem variables restricted to the set of non-active variables.
    Q_U = Q[np.ix_(U, U)] # Q_UU
    A_U = A[:, U] # Matrix of equality constraints
    x_U = x[U,:] # value of x at the current iteration
    g_U = Q_U@x_U + c[U,:] # gradient of the function at x_U

    # Compute Y and Z  

    # Find the permutation matrix. Step 2 in the pseudocode.
    permutation_m = intermfns.find_permutation_matrix(U, partition)    
    
    # Rearrange Au s.t AuP = [I|Az] with P permutation matrix
    AuP = A_U@permutation_m 

    # Compute Y and Z as in step 3. 
    Y = csr_matrix(([1]*k, (list(range(k)), list(range(k)))), shape=(u, k))
    Z = intermfns.compute_Z(AuP, k, u) 

    # Compute py = -h_u = A_ux_u-b. Step 4
    h_U = (A_U@x_U)-b
    py = -h_U

    # Use Cholesky factorization to compute pz. Step 6. 
    # If ZQZ singular status == 0. p will be computed in a different way.
    status, pz = intermfns.compute_pz(Z, permutation_m, Q_U, Y, py, g_U)

    if status == 1: # Cholesky factorization succeed. pz was computed.

        # Compute pu. Step 7
        Y_py = intermfns.matrix_vector_mult(Y, py)
        Z_pz = intermfns.matrix_vector_mult(Z, pz)

        p_U = Y_py + Z_pz
        p_U = permutation_m@p_U
    
        # Compute xu. Step 8
        xu = x_U + p_U 

        if np.all(xu>=-1e-10): # Check feasibility of the new iteration
            
            log('new iter is feasible')
            
            x_ = np.zeros((n,1))
            x_[U,:] = xu

            # Compute Lagrange multipliers lambda. Step  11.
            lag_mult = intermfns.compute_lambdas(permutation_m, Y, g_U, Q_U, p_U)

            # Compute dual multipliers. Step 12.
            mu = intermfns.compute_mult_ineq(Q, x_, c, lag_mult, A) 

            # Find the active variables with  with negative dual multiplier.
            args_ = [(v,mu[v]) for v in np.argwhere(mu <= -1e-10)[:,0] if v in set(L)]

            if len(args_) == 0:    # Check optimality.

                L = np.argwhere(abs(x_)<=-1e-10)[:,0]
                U = [i for i in range(len(x)) if i not in L]
                
                log('objective value: {} \n'.format(0.5*x_.T@Q@x_ + x_.T@c))
                
                return x_, L, U, 'the solution is optimal' # Step 14
            
            else: 

                # Select the variable with the most negative dual multiplier. Step 16
                const_to_drop, mu_v = min(args_, key=lambda t: t[1]) 
                # log('mu chosen: {}'.format(mu_v))
                
                # Update the set of active and non-active variables 
                L = L[L != const_to_drop] # Step 17
                U = np.append(U,const_to_drop) # Step 18
                U.sort()

                log('objective value: {}'.format(0.5*x_.T@Q@x_ + x_.T@c) )
                log('var removed from the active set (L) {} \n'.format(const_to_drop))
                
                return x_, L, U, 'the solution is not optimal'
            
        else :

            log('new iter is not feasible')

            # Define the direction p. Step 22
            p_ = np.zeros((n,1))
            p_[U,:] = p_U 

            # Find the step-length parameter. Step 23, 25.
            cand_alpha = [(-x[i,:]/p_[i,:],i) for i in U if p_[i,:] < -1e-10]
            alpha, min_index = min(cand_alpha, key=lambda t: t[0])
            
            # Update the iteration. Step 24
            x = x+alpha*p_

            # Update the sets of active and non-active variables. Steps 26, 27.
            U = U[U != min_index]
            L = np.append(L,min_index)

            log('objective value: {}'.format(0.5*x.T@Q@x + x.T@c))
            log('step chosen: {}'.format(alpha))
            log('var added to the active set: {} \n'.format(min_index))
            
            return x, L, U, 'the solution is not optimal'

    else: # ZQZ singular. 
        log("Cholesky factorization failed")
        
        # Find p with p in Ker(Q_U) st A_Up=0, <c,p>!=0 and in opposite direction of the gradient
        AU_dense = A_U.toarray() 

        AQ_stack = np.vstack((Q_U, AU_dense))

        # Find an orthogonal basis for the null space of Q_U and A_U.
        Z = null_space(AQ_stack)

        # Project c onto Z.
        proj_q = (c[U,:].T @ Z)*Z
        p_U = np.sum(proj_q, axis=1) 
        p_U = p_U.reshape(-1,1)
        
        # Find p in opposite direction to the gradient
        grad = Q_U@x_U+c[U,:]
        proj_grad =np.sum(((A_U@grad)*AU_dense).T / np.sum(np.square(AU_dense), axis=1), axis=1).reshape(-1,1)
        grad = grad - proj_grad 
        p_U = (p_U.T@grad/abs(p_U.T@grad))*p_U

        if np.sign((c[U,:].T@p_U).flatten()[0]) == 1:
            p_U = -p_U

        # Define the direction p for all the variables.
        p_ = np.zeros((n,1))
        p_[U,:] = p_U 
        
        # Find the step-length parameter. Same as step 23.
        cand_alpha = [(-x[i,:]/p_[i,:],i) for i in U if p_[i,:] < -1e-10]
        alpha, min_index = min(cand_alpha, key=lambda t: t[0])

        # Update the iteration
        x = x+alpha*p_

        # Update the set of active and non-active variables.
        U = U[U != min_index]
        L = np.append(L,min_index)

        log('objective value: {}'.format(0.5*x.T@Q@x + x.T@c))
        log('step chosen: {}'.format(alpha))
        log('const added to the active set: {} \n'.format(min_index))
        
        return x, L, U, 'the solution is not optimal'
            