import numpy as np
import qpsolvers
from qpsolvers import Problem, solve_problem
import quadprog
import time
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import MatricesGenerator as pgenerator
import SolveQP as solveqp

def clarabel_test_function(G, q, A, dim_m, nconst, verbose=True):
    '''
    P (Union[ndarray, csc_matrix]): Symmetric cost matrix (most solvers require it to be definite as well).
    q (ndarray): Cost vector.
    G (Union[ndarray, csc_matrix, None]) : Linear inequality matrix.
    h (Optional[ndarray]) : Linear inequality vector.
    A (Union[ndarray, csc_matrix, None]) : Linear equality matrix.
    b (Optional[ndarray]) : Linear equality vector.
    lb (Optional[ndarray]) : Lower bound constraint vector. Can contain -np.inf.
    ub (Optional[ndarray]) : Upper bound constraint vector. Can contain +np.inf.
    solver (Optional[str]) : Name of the QP solver, to choose in qpsolvers.available_solvers. This argument is mandatory.
    initvals (Optional[ndarray]) : Primal candidate vector  values used to warm-start the solver.
    verbose (bool) : Set to True to print out extra information.
    '''
    start_time = time.process_time()

    # Set the variables of the problem
    P = G
    q = q
    G_g = np.full((dim_m, dim_m), 0)
    h = np.full((1, dim_m), 0).flatten()
    A = A.toarray()
    b = np.ones((1,nconst)).flatten()
    lb = np.array([0]*dim_m)
    ub = np.array([1]*dim_m)

    # Find the objective value
    x = qpsolvers.solve_qp(P, q, G_g, h, A, b, lb, ub, solver='clarabel')
    problem = Problem(P, q, G_g, h, A, b)
    solution = solve_problem(problem, 'clarabel')
    
    # Compute the total execution time
    end_time = time.process_time()
    gpu_time = end_time- start_time

    # Extract additional info of the solution
    extras = solution.extras
    x_extr = solution.x
    found = solution.found

    return x, x_extr, found, solution, gpu_time, extras

def gen_variables_for_tests_except_Q(Q, dim_m, ncons, c_type):

    # Generate the partition that defines the equality constraints
    part = pgenerator.partition(dim_m, ncons)

    # Compute the matrix of constraints A
    A = pgenerator.matrix_of_constraints(part)

    # Define the vector b s.t Ax=b
    b = pgenerator.define_vector_b(ncons)

    # Compute the initial point and the sets of active and non-active constraints
    x0 = pgenerator.initial_point(part)

    # Define the linear part of the problem
    if c_type=='basic':
        c = pgenerator.generate_c_vector(dim_m)
    elif c_type=='special':
        c = pgenerator.generate_special_c_vector(Q,x0)
        
    return c, A, b, x0 

def quadprog_testing_function(G, q, A, b, dim_m, nconst):
    start_time = time.process_time()
    A = np.vstack((A.toarray(), np.identity(dim_m)))
    b = np.vstack((b, np.zeros((dim_m,1))))
    solution = quadprog.solve_qp(G, -q.reshape(-1), A.T, b.reshape(-1), meq=nconst)
    x = solution[0]
    opt_v = solution[1]
    iterations = solution[3]
    end_time = time.process_time()
    quadprog_time = end_time - start_time 
    return x, opt_v, quadprog_time, iterations


def run_exp_and_populate_dict(Q, c, A, b, x0, dim_m, ncons, solver='clarabel', verbose=False, max_iter=2000):
    result = {'n': dim_m, 'k': ncons}
    
    # Run asqp
    opt_v, x_asqp, status, iter, asqp_time = solveqp.active_set_method_optimizer(Q, c, A, b, x0, dim_m, ncons, verbose, max_iter)
    
    result['asqp'] = {"obj_v": opt_v, "x": x_asqp, "status": status, "time": round(asqp_time,4)}
    
    if solver=='clarabel':
        # Run Clarabel
        x, x_extr, found, solution, time_c, extra = clarabel_test_function(Q, c, A, dim_m, ncons)
        status_c = extra['status']
        
        if found==True and x is None:
            x=x_extr
        if found==True:
            obj_v = 0.5*x.T@Q@x + c.T@x
        elif found==False and  x is not None:
            obj_v = 0.5*x.T@Q@x + c.T@x   
        else:
            obj_v = None
        
        result['clarabel'] = {"obj_v": obj_v, "x": x, "status": status_c, "time": time_c, "extra": extra}
    else:
        # Run quadprog
        x_quad, obj_v, time_quadp, iterations = quadprog_testing_function(Q, c, A, b, dim_m, ncons)

        result['quadprog'] = {"obj_v": obj_v, "x": x_quad, "time": time_quadp, "quad_iterations":iterations, "status": 'NA'}

    return result

def print_results(results, solver='clarabel'):

    table_data = []
    
    for res in results:
        if res[solver]['obj_v'] is not None:
            asqp_v = res['asqp']['obj_v']
            solver_v = res[solver]['obj_v']
            # Relative gap
            ref = min(asqp_v, solver_v)
            oth_ = max(asqp_v, solver_v)
            rel_gap = (oth_ - ref)/max(1,abs(ref))

            # Absolute gap
            abs_gap = asqp_v - solver_v
        else:
            rel_gap = None
            abs_gap = None

        table_data.append([res["k"], 
                           res["n"], 
                           res['asqp']['status'], 
                           res['asqp']['obj_v'], 
                           res[solver]['obj_v'], 
                           res['asqp']['time'], 
                           res[solver]['time'],
                           abs_gap, 
                           rel_gap])
        
    headers = ["k", 
               "n",
               "ASQPstat",
               "asqp_sol",
               solver+"_sol",
               "asqp_time",
               solver+"_time",
               "abs_gap",
               "rel_gap"]
    
    print(tabulate(table_data, headers=headers, tablefmt="github"), '\n')