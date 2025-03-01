import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
inputs = np.load('all_inputs.npy', allow_pickle=True)
outputs = np.load('all_outputs.npy')

# Tokenize inputs: Convert '1', '2', '3' to integers 1, 2, 3
tokenized_inputs = [[int(char) for char in sequence] for sequence in inputs]

# Convert inputs to PyTorch tensors
X = torch.tensor(tokenized_inputs, dtype=torch.long)
y = torch.tensor(outputs, dtype=torch.float32)

# Verify shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Create DataLoader for batching
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

torch.save(dataloader, 'dataloader.pth')
torch.save(dataset, 'dataset.pth')

print("Data preprocessed and DataLoader created.")