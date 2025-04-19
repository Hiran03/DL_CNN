# Assignment 2 CNN



A flexible CNN framework for hyperparameter optimization using Weights & Biases sweeps.


## File Descriptions

### 1. `model.py`
Implementation of a **fully configurable CNN** with customizable architecture.



#### Example Usage:
```python
from model import CustomCNN

model = CustomCNN(
    input_shape=(3, 224, 224),
    num_classes=10,
    filters=[64, 64, 64, 64, 64],
    kernel_sizes=[3, 3, 3, 3, 3],
    activations=['silu']*5,
    dense_neurons=128,
    dense_activation='relu',
    dropout_rates=[0, 0, 0.2, 0.2, 0.2],
    use_batchnorm=True,
    pool_sizes=[2, 2, 2, 2, 2]
)

model.train_model(train_loader, val_loader,
                batch_size=32, epochs=10, device='cuda')
```
### 2. `main.ipynb`
Implementation of all the sweeps

### 3. `sweep.py`
Python script to run wandb sweeps. Edit the sweep_config = {} according to your need and run the file.