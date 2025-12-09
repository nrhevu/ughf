import torch
import torch.optim as optim
from models.simple_model import SimpleModel
from losses.base import BaseLoss

class MSELoss(BaseLoss):
    """Example implementation of MSE Loss inheriting from BaseLoss."""
    def forward(self, input, target):
        return torch.mean((input - target) ** 2)

def train():
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 1
    learning_rate = 0.01
    epochs = 5
    
    # Model, Loss, Optimizer
    model = SimpleModel(input_size, hidden_size, output_size)
    criterion = MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Dummy Data
    inputs = torch.randn(5, input_size)
    targets = torch.randn(5, output_size)
    
    print("Starting training loop...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
    print("Training finished.")

if __name__ == '__main__':
    train()
