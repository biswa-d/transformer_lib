# data_loader.py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

# Load tensors
X_tensor = torch.load('X_tensor.pt')
y_tensor = torch.load('y_tensor.pt')

# Define parameters
batch_size = 64
train_split = 0.7
seed = 42  # for reproducibility

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)
dataset_size = len(dataset)
indices = list(range(dataset_size))

# Calculate train and validation sizes
train_size = int(dataset_size * train_split)
valid_size = dataset_size - train_size

# Shuffle the indices
np.random.seed(seed)
np.random.shuffle(indices)

# Split indices into training and validation sets
train_indices, valid_indices = indices[:train_size], indices[train_size:]

# Create samplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

# Create DataLoaders with samplers
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

# Save the DataLoaders for later use
torch.save(train_loader, 'train_loader.pt')
torch.save(val_loader, 'val_loader.pt')
