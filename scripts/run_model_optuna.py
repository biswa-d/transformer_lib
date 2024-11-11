import subprocess

# Step 1: Run data_loader.py to prepare the data
print("Running data_loader.py...")
subprocess.run(["python", "data_loader.py"])

# Step 2: Run transformer_model.py to define and check the model
print("Running transformer_model.py to define the model...")
subprocess.run(["python", "transformer_model.py"])

# Step 3: Run optuna_tuning.py to start hyperparameter tuning with Optuna
print("Running optuna_tuning.py for hyperparameter tuning...")
subprocess.run(["python", "optuna_tuning.py"])

print("Pipeline completed.")
