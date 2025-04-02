import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchvision import transforms
import numpy as np
class CustomCNN(nn.Module):

    def __init__(
        self,
        input_shape=(3, 224, 224),  # (channels, height, width)
        num_classes=10,
        filters=[32, 64, 128, 256, 512],  # Filters per conv block
        kernel_sizes=[3, 3, 3, 3, 3],      # Kernel sizes per block
        activations=['relu', 'relu', 'relu', 'relu', 'relu'],  # Conv block activations
        dense_neurons=128,
        dense_activation='relu',
        dropout_rates=[0.1, 0.1, 0.2, 0.2, 0.3],  # Dropout after each conv block
        use_batchnorm=True,               # BatchNorm after each conv
        pool_sizes=[2, 2, 2, 2, 2]        # Maxpool kernel size per block
    ):
        super().__init__()
        
        
        self.activations = activations
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batchnorms = nn.ModuleList() if use_batchnorm else None
        
        in_channels = input_shape[0]
        
        # Build conv blocks
        for i in range(5):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                padding=kernel_sizes[i]//2  # Same padding
            )
            self.conv_blocks.append(conv)
            
            if use_batchnorm:
                self.batchnorms.append(nn.BatchNorm2d(filters[i]))
                
            self.pools.append(nn.MaxPool2d(kernel_size=pool_sizes[i]))
            self.dropouts.append(nn.Dropout(dropout_rates[i]))
            in_channels = filters[i]
            
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            for i in range(5):
                dummy = self.conv_blocks[i](dummy)
                if use_batchnorm:
                    dummy = self.batchnorms[i](dummy)
                dummy = self._apply_activation(activations[i], dummy)
                dummy = self.pools[i](dummy)
            flattened_size = dummy.numel()
            
        # Dense layers
        self.fc1 = nn.Linear(flattened_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, num_classes)
        self.dense_activation = dense_activation
        
    def _apply_activation(self, name, x):
        if name == 'relu':
            return F.relu(x)
        elif name == 'leaky_relu':
            return F.leaky_relu(x, 0.1)
        elif name == 'sigmoid':
            return torch.sigmoid(x)
        elif name == 'gelu':
            return F.gelu(x)  # Gaussian Error Linear Unit
        elif name == 'silu' or name == 'swish':
            return F.silu(x)  # Sigmoid-Weighted Linear Unit (SiLU/Swish)
        elif name == 'mish':
            # Mish: x * tanh(softplus(x))
            return x * torch.tanh(F.softplus(x))
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        for i in range(5):
            x = self.conv_blocks[i](x)
            if self.batchnorms is not None:
                x = self.batchnorms[i](x)
            x = self._apply_activation(self.activations[i], x)
            x = self.pools[i](x)
            x = self.dropouts[i](x)
            
        x = torch.flatten(x, 1)
        x = self._apply_activation(self.dense_activation, self.fc1(x))
        x = self.fc2(x)  # No softmax (handled in loss)
        return x
    
    def train_model(self, train_loader, val_loader, 
               batch_size=32, epochs=10, lr=0.001, device='cpu'):
        """
        Train the model with proper device handling and training loop
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            batch_size: Size of mini-batches
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on ('cpu' or 'cuda')
        """
        # Move model to device first
        self.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Wrap train_loader with tqdm for progress bar
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_X, batch_y in train_iter:
                # Move data to same device as model
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Calculate statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # Update progress bar
                train_iter.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * correct / total
                })
            
            # Calculate epoch statistics
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader, device)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
    def evaluate(self, loader, device='cpu'):
        """
        Evaluate the model on a given dataset
        
        Args:
            loader: DataLoader for evaluation data
            device: Device to evaluate on ('cpu' or 'cuda')
        
        Returns:
            tuple: (average loss, accuracy percentage)
        """
        # Set device if not specified
        if device is None:
            device = next(self.parameters()).device
            
        # Switch to evaluation mode
        self.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                # Move data to the same device as model
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = self(batch_X)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)  # Weight by batch size
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, loader, device=None):
        """
        Make predictions using a DataLoader.
        
        Args:
            loader: DataLoader containing input data
            device: Target device ('cpu', 'cuda', or None for auto-detection)
        
        Returns:
            numpy.ndarray: Array of predicted class indices
        """
        # Set model to eval mode
        self.eval()
        
        # Auto-detect device if not specified
        if device is None:
            device = next(self.parameters()).device
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                # Handle both (data,) and (data, target) batch formats
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(device)
                
                outputs = self(inputs)
                _, batch_preds = torch.max(outputs, 1)
                predictions.append(batch_preds.cpu().numpy())
        
        return np.concatenate(predictions)
