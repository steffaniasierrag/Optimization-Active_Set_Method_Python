import numpy as np
import HelperFunctions as intermfns
import ActiveSet as as_solver
import time

def active_set_method_optimizer(Q, c, A, b, x0, n, k, verbose, max_iter=2000):
  """Solve a convex quadratic program:
    minimize    1/2 x^T Q x + x^T q
    subject to  sum_{i in I^k}x_i = 1, for k in K
                x >= 0
    where K is a set of disjoint simplices, and I^k form a partition of the set {1,...,n}.

    Args:
        Q: nxn symmetric positive semidefinite matrix, the matrix Q from above.
        c: nx1 vector representing the linear part of the quadratic problem.
        A: kxn matrix representing the equality constraints.
        b: kx1 constraint vector from above.
        x0: nx1 initial feasible vector.
    
    Returns:
        v: the best real scalar function value found so far, possibly, the optimal one.
        x: nx1 best solution vector found so far. 
        status: a string describing the status of the algorithm at termination, with the following possible values:
                'optimal': the algorithm terminated having proven that x is an
                approximately optimal solution.
                'stopped': the algorithm terminated having exhausted the maximum
                number of iterations. x is the best solution found so far, but not
                necessarily the optimal one.
  """
  # Record the start time for timing the program
  start_time = time.process_time()

  if Q.shape[0] != Q.shape[1]:
     raise ValueError('Q must be a square matrix. Receive shape=({},{})'.format(Q.shape[0], Q.shape[1]))
  if c.shape[0] != Q.shape[0]:
     raise ValueError('Q and c must have the same dimension. Received Q as {} and c as {}'.format(Q.shape, c.shape))
  if Q.shape[0] != A.shape[1]:
     raise ValueError('Q and A.T must have the same first dimension. Received Q as {} and A as {}'.format(Q.shape, A.shape))
  if b.shape[0] != A.shape[0]:
    raise ValueError('The number of rows of A must match the length of b. Received A as {} and b as {}'.format(A.shape, b.shape))

  partition = intermfns.find_the_partition(A)

  n = x0.shape[0]
  k = A.shape[0]

  # Starting variables
  x = x0
  U = np.nonzero(x)[0]
  L = np.argwhere(x == 0)[:, 0]
  
  res = 'the solution is not optimal'
  iter = 0

  while res == 'the solution is not optimal' and iter <= max_iter:    
    x, L, U, res = as_solver.activesetmethod2(Q, c, A, b, x, L, U, n, k, partition, iter, verbose)
    iter +=1

  # Compute the objective value reached
  opt_v = 0.5*x.T@Q@x + x.T@c
  opt_v = opt_v.flatten()[0]

  # Define the final status of the output returned.
  if res == 'the solution is not optimal' and iter > max_iter: 
    status = 'stopped'
  else: 
    status = 'optimal' 

  # Compute the total execution time
  end_time = time.process_time()
  our_algo_time = end_time - start_time

  return opt_v, x, status, iter, our_algo_time
