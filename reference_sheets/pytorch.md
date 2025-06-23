# 🧠 PyTorch Cheat Sheet: Code Snippets + Inline Comments

## Importing PyTorch Modules
Basic imports you'll need for neural network models, datasets, and utilities.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
```

## Tensor Basics
How to create tensors and move them to GPU if available.
```python
x = torch.tensor([1.0, 2.0, 3.0])           # 1D tensor
x = torch.rand(2, 3)                        # 2x3 random tensor
x = x.cuda() if torch.cuda.is_available() else x  # Move to GPU if available
```

## Autograd Example
How PyTorch computes gradients using automatic differentiation.
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()                               # Compute gradient
grad = x.grad                              # dy/dx = 2x = 4
```

## Simple Feedforward Neural Network
How to define a basic neural network using `nn.Module`.
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))            # ReLU activation
        return self.fc2(x)                 # Output layer

model = Net()
```

## Optimizer and Loss Function
Setup for model optimization.
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
```

## Dummy Dataset and DataLoader
Create synthetic data and use `DataLoader` for batching.
```python
x_train = torch.rand(100, 10)
y_train = torch.rand(100, 1)
dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

## Training Loop
Standard loop to train a PyTorch model.
```python
for epoch in range(10):
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)                  # Forward pass
        loss = loss_fn(y_pred, y_batch)          # Compute loss
        optimizer.zero_grad()                    # Clear old gradients
        loss.backward()                          # Backprop
        optimizer.step()                         # Update weights
```

## Evaluation Mode
Use `model.eval()` and `torch.no_grad()` for inference.
```python
model.eval()                                     # Disable dropout/batchnorm
with torch.no_grad():
    preds = model(x_train[:5])
```

## Save and Load Model
How to persist and reload model weights.
```python
torch.save(model.state_dict(), 'model.pth')     # Save weights
model.load_state_dict(torch.load('model.pth'))  # Load weights
model.eval()
```

## Useful Tricks
Quick utilities for checking CUDA and model device.
```python
print(torch.cuda.is_available())                # Check for GPU
print(next(model.parameters()).device)          # Where the model is (CPU/GPU)
```

## Loss Functions Overview
Common loss functions and when to use them.
```python
# nn.BCELoss()             # Binary classification
# nn.CrossEntropyLoss()    # Multi-class classification (uses logits)
# nn.MSELoss()             # Regression
```

## Common Layers
Useful layers for different model architectures.
```python
# nn.Linear(in_features, out_features)
# nn.Conv2d(in_channels, out_channels, kernel_size)
# nn.LSTM(input_size, hidden_size, num_layers)
```

## Model Modes
How to toggle between training and evaluation behavior.
```python
# model.train()            # Training mode (enables dropout, BN updates)
# model.eval()             # Evaluation mode (disables dropout, BN updates)
```

## Gradient Debugging
Inspect gradients to verify backprop.
```python
# for name, param in model.named_parameters():
#     print(name, param.grad)     # View gradients after loss.backward()
```

## GPU Debugging
How to explicitly move data and model to GPU.
```python
# x = x.to('cuda')
# model = model.to('cuda')
# y_pred = model(x)
```
