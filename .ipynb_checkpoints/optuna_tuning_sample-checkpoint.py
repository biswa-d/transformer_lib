# optuna_tuning.py
import optuna
import torch
import torch.optim as optim
from transformer_model import TransformerModel
from training_validation import train_model, validate_model
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def objective(trial):
    # Suggest hyperparameters
    num_heads = trial.suggest_int("num_heads", 1, 2)
    # Make embed_size divisible by num_heads
    embed_size = num_heads * trial.suggest_int("embed_factor", 2, 4)  # Factors of num_heads
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    e_num_layers = trial.suggest_int("e_num_layers", 1, 4)
    d_num_layers = trial.suggest_int("d_num_layers", 1, 4)
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.3)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3)

    print(f"Starting trial with params: num_heads={num_heads}, embed_size={embed_size}, "
          f"hidden_size={hidden_size}, e_num_layers={e_num_layers}, d_num_layers={d_num_layers}, "
          f"dropout_prob={dropout_prob}, lr={learning_rate}")

    # Initialize model and optimizer
    model = TransformerModel(
        input_size=3,
        output_size=1,
        embed_size=embed_size,
        hidden_size=hidden_size,
        e_num_layers=e_num_layers,
        d_num_layers=d_num_layers,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        device=device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data loaders
    train_loader = torch.load('train_loader.pt')
    val_loader = torch.load('val_loader.pt')

    for epoch in range(2):  # Adjust epochs as needed
        print(f"Epoch {epoch+1}/2 for current trial")

        # Training phase
        train_loss = train_model(model, train_loader, optimizer)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation phase
        val_loss = validate_model(model, val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Report to Optuna and check for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print("Trial pruned due to insufficient performance.")
            raise optuna.TrialPruned()

    print(f"Trial completed with Validation Loss: {val_loss:.4f}")
    return val_loss

print("Starting Optuna hyperparameter tuning...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Save the best parameters found
print("Saving best hyperparameters to best_params.pkl")
with open("best_params.pkl", "wb") as f:
    pickle.dump(study.best_params, f)

print("Best hyperparameters:", study.best_params)
