
import scipy.io
from scipy.sparse import csr_matrix
import random
import numpy as np

import MatricesGenerator as pgenerator
import FunctionsForExperiments as testfns

SEED = 13
random.seed(SEED) 
np.random.seed(SEED)

sp_sizes = [100, 200, 500, 1000]
k_values = [10, 50, 90]

# Experiments 1 & 2: Q full rank

## Basic case: q random
print("Experiment 1: Q positive semidefinite matrix and full rank, and q random")
exp_numb = 1
results_1 = []

for k in k_values:
    for n in sp_sizes:
        print('K =:',str(k),'Size = ',str(n))
        Q = pgenerator.generate_symmetric_psd_matrix(n,n)
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'basic')
        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k)
        results_1.append(result)

testfns.print_results(results_1)

## Special case: q=-Qx0
print("Experiment 2: Q positive semidefinite matrix and full rank, and q= -Qx0")
exp_numb = 2
results_2 = []

for k in k_values:
    for n in sp_sizes:
        print('K =:',str(k),'Size = ',str(n))
        Q = pgenerator.generate_symmetric_psd_matrix(n,n)
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'special')
        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k)
        results_2.append(result)

testfns.print_results(results_2)


# Experiments 3 & 4: Q low rank dense matrix

## Basic case: q random
print("Experiment 3: Q positive semidefinite matrix and low rank (rank=50), and q random")

exp_numb = 3
results_3 = []

for k in k_values:
    for n in sp_sizes:
        print('K =:',str(k),'Size = ',str(n))
        Q = pgenerator.generate_symmetric_psd_matrix(n,50)
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'basic')

        max_iter = 2000
        if n>=500:
            max_iter = 5000

        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k, 'clarabel', verbose=False, max_iter=max_iter)  
        results_3.append(result)

testfns.print_results(results_3)

## Special case: q=-Qx0
print("Experiment 4:  Q positive semidefinite matrix and low rank (rank=50), and q= -Qx0")
exp_numb = 4
results_4 = []
for k in k_values:
    for n in sp_sizes:
        print('K =:',str(k),'Size = ',str(n))
        Q = pgenerator.generate_symmetric_psd_matrix(n,50)
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'special')

        max_iter = 2000
        if n>=500:
            max_iter = 5000

        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k, 'clarabel', verbose=False, max_iter=max_iter)   
        results_4.append(result)

testfns.print_results(results_4)

# Experiments 5 & 6: Q sparse

## Basic case: q random
print("Experiment 5: Q positive semidefinite, full rank and sparse, and q basic")

exp_numb = 5
results_5 = []
for k in k_values:
    for i in sp_sizes:
        print('K =:',str(k),'Size = ',str(n))
        mat = scipy.io.loadmat('C:/Users/SteffaniaSierraG/Documents/ODS_PROJECT/active_set_method_code/ActiveSetAlgorithm/SparseMatrices/matrix'+str(n)+'spar0.2.mat')
        sparse_matlab = mat['sparse_psd']
        sparse_python = csr_matrix(sparse_matlab)
        Q = sparse_python.toarray()
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'basic')
        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k)  
        results_5.append(result)

testfns.print_results(results_5)

## Special case: q=-Qx0
print("Experiment 6: Q positive semidefinite, full rank and sparse, and q= -Qx0")
exp_numb = 6
results_6 = []
for k in k_values:
    for n in sp_sizes:
        print('K =:',str(k),'Size = ',str(n))
        mat = scipy.io.loadmat('C:/Users/SteffaniaSierraG/Documents/ODS_PROJECT/active_set_method_code/ActiveSetAlgorithm/SparseMatrices/matrix'+str(n)+'spar0.2.mat')
        sparse_matlab = mat['sparse_psd']
        sparse_python = csr_matrix(sparse_matlab)
        Q = sparse_python.toarray()
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'special')
        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k)
        results_6.append(result)
            
testfns.print_results(results_6)

# Experiments 7 & 8: Q positive definite

## Basic case: q random
print("Experiment 7: Q positive definite, full rank and q random")
exp_numb = 7
results_7 = []
for k in k_values:
    for n in sp_sizes:
        print(f"K =: {k} Size = {n}")
        Q = pgenerator.generate_symmetric_pd_matrix(n)
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'basic')
        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k, 'quadprog')
        results_7.append(result)

testfns.print_results(results_7, 'quadprog')

## Special case: q=-Qx0
print("Experiment 8: Q positive definite, full rank and q= -Qx0")
exp_numb = 8
results_8 = []
for k in k_values:
    for n in sp_sizes:
        print(f"K =: {k} Size = {n}")
        Q = pgenerator.generate_symmetric_pd_matrix(n)
        c, A, b, x0 = testfns.gen_variables_for_tests_except_Q(Q, n, k, 'special')
        result = testfns.run_exp_and_populate_dict(Q, c, A, b, x0, n, k, 'quadprog')
        results_8.append(result)

testfns.print_results(results_8, 'quadprog') 