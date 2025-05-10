import numpy as np
import itertools

def create_matrices(n, a, b):
    A = np.eye(n)
    B = np.eye(n)
    
    for i in range(n - 1):
        A[i, i + 1] = a
        B[i + 1, i] = b
    
    return A, B

def matrix_power(matrix, l):
    return np.linalg.matrix_power(matrix, l)

def inverse_matrix(matrix):
    return np.linalg.inv(matrix)

def compute_hash_function(sequence, matrices, p):
    result = np.eye(matrices[1].shape[0])
    for char in sequence:
        number = int(char)
        result = np.dot(result, matrices[number])
    return np.mod(result, p)

# Parameters
n = 3
a = 4
b = 2
l = 4
p = 5

# Create the matrices
A, B = create_matrices(n, a, b)
A_power_l = matrix_power(A, l)
B_power_l = matrix_power(B, l)
A_inv = inverse_matrix(A_power_l)
B_inv = inverse_matrix(B_power_l)

s = {
    1: B_power_l,
    2: A_inv,
    3: B_inv
}

# Generate all possible sequences of length 8
sequences = [''.join(seq) for seq in itertools.product('123', repeat=8)]

# Compute the corresponding hash function outputs
inputs = []
outputs = []
for sequence in sequences:
    output_matrix = compute_hash_function(sequence, s, p)
    inputs.append(sequence)
    outputs.append(output_matrix.flatten())

inputs = np.array(inputs)
outputs = np.array(outputs)

# Save the dataset
np.save('all_inputs.npy', inputs)
np.save('all_outputs.npy', outputs)

print("All possible sequences generated and saved.")