import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   batch_size=32, epochs=10, lr=0.001, device='cpu'):
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.LongTensor(y_val).to(device)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Move model to device
        self.to(device)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Print batch stats
                print(f"\rBatch loss: {loss.item():.4f}", end='')
            
            # Epoch stats
            train_loss = running_loss / len(train_loader)
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size, device)
            
            print(f"\nEpoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc:.2f}%")
    
    def evaluate(self, X, y, batch_size=32, device='cpu'):
        if device is None:
            device = next(self.parameters()).device  # Use model's current device
        self.eval()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(loader), 100 * correct / total
    
    def predict(self, X, batch_size=32, device=None):
        """
        Make predictions on input data.
        
        Args:
            X: Input data (numpy array or torch.Tensor)
            batch_size: Batch size for prediction
            device: Target device ('cpu', 'cuda', or None for auto-detection)
        
        Returns:
            List of predicted class indices
        """
        # Set model to eval mode
        self.eval()
        
        # Auto-detect device if not specified
        if device is None:
            device = next(self.parameters()).device
        
        # Convert input to tensor and move to correct device
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(device)
        
        # Create DataLoader without unnecessary copying
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                # Handle single-tensor batches properly
                inputs = batch[0].to(device)
                outputs = self(inputs)
                _, batch_preds = torch.max(outputs, 1)
                predictions.extend(batch_preds.cpu().numpy().tolist())
        
        return predictions
