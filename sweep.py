import wandb
from model import CustomCNN  # Assuming your model class is named CustomCNN
import numpy as np

# Correct sweep configuration
sweep_config = {
    "name": "CNN_Sweep_v1",
    "method": "grid",  # Alternatives: "random" or "bayes"
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [5]
        },
        "filters": {
            "values": [
                [32, 32, 32, 32, 32],
                [32, 64, 128, 256, 512]
            ]
        },
        "activations": {
            "values": [
                ['relu']*5,
                ['selu']*5
            ]
        },
        "use_batchnorm": {
            "values": [True, False]
        },
        "dropout_rates": {
            "values": [
                [0.1, 0.1, 0.2, 0.2, 0.3],
                [0.2, 0.2, 0.3, 0.3, 0.4]
            ]
        },
        "learning_rate": {
            "values": [0.001, 0.0001]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="DL_CNN")

def train():
    # Initialize W&B run
    run = wandb.init(project="DL_CNN")
    
    # Generate descriptive run name
    run_name = (f"filters_{'_'.join(map(str, wandb.config.filters))}_"
                f"act_{wandb.config.activations[0]}_"
                f"bn_{wandb.config.use_batchnorm}_"
                f"dr_{'_'.join(map(str, wandb.config.dropout_rates))}")
    wandb.run.name = run_name
    
    # Initialize model with sweep parameters
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        filters=wandb.config.filters,
        kernel_sizes=[3, 3, 3, 3, 3],
        activations=wandb.config.activations,
        dense_neurons=128,
        dense_activation='relu',
        dropout_rates=wandb.config.dropout_rates,
        use_batchnorm=wandb.config.use_batchnorm,
        pool_sizes=[2, 2, 2, 2, 2]
    )
    
    # Train the model (assuming your model has these methods)
    model.train(X_train, y_train, epochs=wandb.config.epochs, batch_size=32)
    
    # Evaluate
    y_pred = model.predict(X_val)
    val_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1))
    
    # Log metrics
    wandb.log({
        "val_accuracy": val_accuracy,
        "epochs": wandb.config.epochs
    })
    
    run.finish()

# Start the sweep
wandb.agent(sweep_id, train)