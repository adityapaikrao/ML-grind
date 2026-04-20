import torch 
import torch.nn.functional as F
from typing import Tuple


class NeuralNet:
    """
    Implements a Simple 2-layer Neural Network. For classification task. 
    """
    def __init__(self,
                 d_in: int,
                 d1: int,
                 d2: int,
                 d_out: int,
                 learning_rate: float = 1e-4
            ):
        
        self.W1 = torch.randn(d_in, d1) * 0.01
        self.B1 = torch.randn(1, d1) * 0.01
        self.activation1 = F.relu
        
        self.W2 = torch.randn(d1, d2) * 0.01
        self.B2 = torch.randn(1, d2) * 0.01
        self.activation2 = F.relu
        
        self.W3 = torch.randn(d2, d_out) * 0.01
        self.B3 = torch.randn(1, d_out) * 0.01
        self.activation3 = F.sigmoid

        self.loss = F.binary_cross_entropy
        self.learning_rate = learning_rate
    
    def forward(self, x: torch.Tensor, grad=False) -> torch.Tensor:
        """
        Forward Pass for our 2 layer network
        """
        # layer 1
        self.Z1 = x @ self.W1 + self.B1 # N, d1
        self.A1 = self.activation1(self.Z1) # N, d1

        # layer 2 
        self.Z2 = self.A1 @ self.W2 + self.B2 # N, d2
        self.A2 = self.activation2(self.Z2) # N, d2

        # layer 3
        Z3 = self.A2 @ self.W3 + self.B3 # N, dout
        Y = self.activation3(Z3)

        if grad:
            self.W3_cache = self.W3
            self.W2_cache = self.W2
            self.x_cache = x


        return Y

    def backprop(self, yhat: torch.Tensor, y: torch.Tensor) -> None:
        """
        Performs the backpropagation stage for the network. Updates the Parameters.
        """

        # compute error
        upstream3 = yhat - y # N, dout

        # layer 3 update
        dW3 = self.A2.T @ upstream3  # input * activation_grad * error
        dB3 = torch.sum(upstream3, dim=0, keepdim=True)

        self.W3 -= self.learning_rate * dW3
        self.B3 -= self.learning_rate * dB3

        # layer 2 update
        upstream2 = (upstream3 @ self.W3_cache.T) * (self.Z2 > 0)
        dW2 = self.A1.T @ upstream2
        dB2 = torch.sum(upstream2, dim=0, keepdim=True)

        self.W2 -= self.learning_rate * dW2
        self.B2 -= self.learning_rate * dB2


        # layer 1 update
        upstream1 = (upstream2 @ self.W2_cache.T) * (self.Z1 > 0)
        dW1 = self.x_cache.T @ upstream1
        dB1 = torch.sum(upstream1, dim=0, keepdim=True)

        self.W1 -= self.learning_rate * dW1
        self.B1 -= self.learning_rate * dB1

        return
    

if __name__ == "__main__":
    # Test custom NN on a synthetic binary classification dataset
    torch.manual_seed(42)

    # Create two Gaussian blobs in 2D
    n_per_class = 500
    class0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, -2.0])
    class1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 2.0])

    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([
        torch.zeros(n_per_class, 1),
        torch.ones(n_per_class, 1)
    ], dim=0)

    # Shuffle and split
    indices = torch.randperm(X.shape[0])
    X = X[indices]
    y = y[indices]

    split = int(0.8 * X.shape[0])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize model
    model = NeuralNet(d_in=2, d1=16, d2=8, d_out=1, learning_rate=1e-2)

    # Train
    epochs = 1000
    for epoch in range(epochs):
        yhat = model.forward(X_train, grad=True)
        loss = model.loss(yhat, y_train)
        model.backprop(yhat, y_train)

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                preds = (yhat >= 0.5).float()
                acc = (preds == y_train).float().mean().item()
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f} Train Acc: {acc:.4f}")

    # Evaluate
    with torch.no_grad():
        yhat_test = model.forward(X_test)
        test_loss = model.loss(yhat_test, y_test).item()
        test_preds = (yhat_test >= 0.5).float()
        test_acc = (test_preds == y_test).float().mean().item()

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")