from torchvision import transforms
import wandb
from model import CustomCNN 
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms, datasets

# Training transforms (with augmentation)
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Base transform (without augmentation)
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Required sweep configuration
sweep_config = {
    "name": "CNN_Sweep_v3",
    "method": "grid",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [10]
        },
        "filters": {
            "values": [
                [64, 64, 64, 64, 64]
            ]
        },
        "activations": {
            "values": [
                ['silu']*5
            ]
        },
        "use_batchnorm": {
            "values": [True]
        },
        "dropout_rates": {
            "values": [
                [0, 0, 0.2, 0.2, 0.2]
            ]
        },
        "augmentation": {
            "values": [True, False]
        }
    }
}

def train():
    run = wandb.init(project="DL_CNN", reinit=True)
    
    # Load datasets
    data_dir = "DL_ASS2/inaturalist_12K/train"
    
    # Always use base transform for validation (20% split)
    full_dataset = datasets.ImageFolder(data_dir, transform=base_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply augmentation to training set if configured
    if wandb.config.augmentation:
        aug_dataset = datasets.ImageFolder(data_dir, transform=aug_transform)
        # Combine original and augmented training data
        train_dataset = ConcatDataset([train_dataset, aug_dataset])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model
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
    
    # Train and evaluate
    model.train_model(train_loader, val_loader,
                    batch_size=8, epochs=wandb.config.epochs, device='cuda')
    acc = model.evaluate(val_loader, device="cuda")[1]
    wandb.log({"val_accuracy": acc})
    run.finish()

# Initialize and run sweep
sweep_id = wandb.sweep(sweep_config, project="DL_CNN")
wandb.agent(sweep_id, function=train)