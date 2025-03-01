"""
The following hash function is based of the paper "Post-quantum hash functions using SLn(Fp)" 
by Corentin Le Coz∗, Christopher Battarbee, Ram´ on Flores†,Thomas Koberda‡, and Delaram Kahrobaei§
We use details given in section 2.3's general construction followed by an example of the 
function using the same sequence given in 2.4's concrete example. The program uses the
helper functions defined by s1(1) = B, s1(2) = -1, s1(3) = B-1.

"""
import numpy as np

def create_matrices(n, a, b):
    # Create an identity matrix of size n x n for A
    A = np.eye(n)
    # Create an identity matrix of size n x n for B
    B = np.eye(n)
    
    for i in range(n - 1):
        # Loop to set the value 'a' above the main diagonal in matrix A
        A[i, i + 1] = a
        # Loop to set the value 'b' below the main diagonal in matrix B
        B[i + 1, i] = b
    
    return A, B

def matrix_power(matrix, l):
    return np.linalg.matrix_power(matrix, l)

def inverse_matrix(matrix):
    return np.linalg.inv(matrix)

def compute_hash_function(sequence, matrices):
    result = np.eye(matrices[1].shape[0])  # Start with the identity matrix
    # Iterate over each number in the input sequence
    for number in sequence:
        # Multiply the current result by the matrix corresponding to the current number
        result = np.dot(result, matrices[number])
    # Return the final result after processing the entire sequence
    return result

# Parameters
n = 3  # Size of the matrix
a = 4  # Value for A
b = 2  # Value for B
l = 4  # Power to which the matrices are raised

# Create the matrices
A, B = create_matrices(n, a, b)

# Raise the matrices to the power l
A_power_l = matrix_power(A, l)
B_power_l = matrix_power(B, l)

# Compute their inverses
A_inv = inverse_matrix(A_power_l)
B_inv = inverse_matrix(B_power_l)

# Create the mapping
s = {
    1: B_power_l,
    2: A_inv,
    3: B_inv
}

# Example sequence
sequence = [2, 2, 3, 2, 2, 2, 1]  

# Compute the result of the hash function
result = compute_hash_function(sequence, s)

print("Result of the hash function:")
print(result)

"""
The result should be:
Result of the hash function:
[[ 6.94190977e+08  2.33260720e+08  2.92979520e+07]
 [-3.83796480e+07 -1.28962550e+07 -1.61979200e+06]
 [ 1.19193600e+06  4.00512000e+05  5.03050000e+04]]
 
These numbers are equal to the ones given by the concrete example in section 2.
"""