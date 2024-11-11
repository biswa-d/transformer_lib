# training_validation.py
import torch
import torch.nn.functional as F
from transformer_model import TransformerModel

def train_model(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        print(f"Processing batch {i+1}/{len(train_loader)}")  # Add batch progress logging

        X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
        optimizer.zero_grad()
        
        loss = model.training_step(X_batch, y_batch, optimizer)
        total_loss += loss * X_batch.size(0)
        
        print(f"Batch {i+1} Loss: {loss:.4f}")  # Add batch loss logging

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(model.device), y_val.to(model.device)
            loss = model.validation_step(X_val, y_val)
            val_loss += loss * X_val.size(0)
    return val_loss / len(val_loader.dataset)
