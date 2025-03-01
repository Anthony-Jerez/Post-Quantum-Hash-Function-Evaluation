import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class HashFunctionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(HashFunctionPredictor, self).__init__()
        self.embedding = nn.Embedding(4, 10)  # 4 tokens (0-3) with embedding size 10
        self.lstm = nn.LSTM(10, 50, batch_first=True)
        self.fc = nn.Linear(50 * input_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# Load DataLoader and dataset
dataset = torch.load('dataset.pth')
X, y = dataset.tensors
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Parameters
input_size = X.shape[1]
output_size = y.shape[1]

# Split the dataset into training and testing sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, loss function, and optimizer
model = HashFunctionPredictor(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement before stopping
best_test_loss = float('inf')
patience_counter = 0

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')
    
    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    test_loss /= len(test_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss}')
    
    # Save the best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model improved and saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered. Training halted.")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
print("Best model loaded.")

# Function used to predict the input sequence from the specified output vector
def predict_input(output_vector, model, X):
    model.eval()
    with torch.no_grad():
        output_vector = output_vector.unsqueeze(0)  # Add batch dimension
        predictions = model(X)
        closest_index = torch.argmin(torch.norm(predictions - output_vector, dim=1))
        predicted_sequence = X[closest_index]
        return ''.join(map(str, predicted_sequence.numpy())), closest_index

# Randomly select an output vector from the test dataset
test_dataset_indices = list(range(len(test_dataset)))
random_index = np.random.choice(test_dataset_indices)
random_output_vector = test_dataset[random_index][1]
random_output_tensor = torch.tensor(random_output_vector.clone().detach(), dtype=torch.float32)

# Print the randomly selected output matrix and its index
print("Randomly selected output matrix (flattened):")
print(random_output_vector)
print("Index of the randomly selected output matrix:", random_index)

# Predict and print the input sequence based on the randomly selected output matrix after training
predicted_input, closest_index = predict_input(random_output_tensor, model, X)
print("Predicted input sequence:")
print(predicted_input)

# Print the actual original input sequence corresponding to the randomly selected output matrix
actual_input_sequence = ''.join(map(str, test_dataset[random_index][0].numpy()))
print("Actual input sequence for the randomly selected output matrix:")
print(actual_input_sequence)