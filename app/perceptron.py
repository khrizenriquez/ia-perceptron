"""
Module implementing a simple perceptron for binary classification.
"""
from typing import List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


class Perceptron:
    """
    Implementation of a simple perceptron for binary classification.
    
    Attributes:
        weights (np.ndarray): Synaptic weights of the perceptron.
        bias (float): Activation threshold (bias).
        learning_rate (float): Learning rate for weight adjustment.
        max_epochs (int): Maximum number of training epochs.
        weights_history (List): History of weights during training.
        bias_history (List): History of biases during training.
        error_history (List): History of errors during training.
    """
    
    def __init__(self, 
                 input_size: int,
                 weights: Optional[np.ndarray] = None,
                 bias: float = 0.0,
                 learning_rate: float = 0.1,
                 max_epochs: int = 100):
        """
        Initialize the perceptron with configurable values.
        
        Args:
            input_size: Dimension of input data.
            weights: Initial weights (if None, randomly generated).
            bias: Threshold value (bias).
            learning_rate: Learning rate η.
            max_epochs: Maximum number of training epochs.
        """
        if weights is None:
            self.weights = np.random.randn(input_size)
        else:
            self.weights = np.array(weights)
            
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # History for training tracking
        self.weights_history = [self.weights.copy()]
        self.bias_history = [self.bias]
        self.error_history = []
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the given inputs.
        
        Args:
            X: Input data matrix, where each row is a sample.
        
        Returns:
            Binary predictions (0 or 1) for each sample.
        """
        # Calculate weighted sum of inputs and weights
        activation = np.dot(X, self.weights) + self.bias
        
        # Apply step activation function
        return np.where(activation >= 0, 1, 0)
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train the perceptron for a complete epoch.
        
        Args:
            X: Input data matrix.
            y: Target label vector (0 or 1).
            
        Returns:
            Total error in the current epoch.
        """
        error_count = 0
        
        # Iterate over each training sample
        for i in range(len(X)):
            # Get current prediction
            prediction = self.predict(X[i].reshape(1, -1))[0]
            
            # Calculate error
            error = y[i] - prediction
            
            # Update error counter
            if error != 0:
                error_count += 1
                
                # Update weights and bias according to the perceptron rule
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
        
        # Save weight history
        self.weights_history.append(self.weights.copy())
        self.bias_history.append(self.bias)
        
        # Calculate error rate (proportion of misclassified samples)
        error_rate = error_count / len(X)
        self.error_history.append(error_rate)
        
        return error_rate
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[float], int]:
        """
        Train the perceptron until convergence or reaching max_epochs.
        
        Args:
            X: Input data matrix.
            y: Target label vector (0 or 1).
            
        Returns:
            Tuple with (error history, number of epochs performed).
        """
        # Reset histories
        self.weights_history = [self.weights.copy()]
        self.bias_history = [self.bias]
        self.error_history = []
        
        # Training by epochs
        epoch = 0
        error_rate = 1.0  # Initialize with maximum error
        
        while error_rate > 0 and epoch < self.max_epochs:
            error_rate = self.train_epoch(X, y)
            epoch += 1
            
            # If error is zero, perceptron has converged
            if error_rate == 0:
                break
                
        return self.error_history, epoch
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                              title: str = "Frontera de Decisión") -> plt.Figure:
        """
        Generate a plot with the perceptron's decision boundary (only for 2D).
        
        Args:
            X: Input data (only works for 2D data).
            y: True labels.
            title: Plot title.
            
        Returns:
            Matplotlib figure with the decision boundary.
        """
        if X.shape[1] != 2:
            raise ValueError("Boundary visualization only works with 2D data")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points
        ax.scatter(X[y==0, 0], X[y==0, 1], label='Clase 0', marker='o')
        ax.scatter(X[y==1, 0], X[y==1, 1], label='Clase 1', marker='x')
        
        # Calculate decision boundary
        if self.weights[1] != 0:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min = (-self.bias - self.weights[0] * x_min) / self.weights[1]
            y_max = (-self.bias - self.weights[0] * x_max) / self.weights[1]
            ax.plot([x_min, x_max], [y_min, y_max], 'k-', label='Frontera de Decisión')
        else:
            # Case of vertical boundary
            x_boundary = -self.bias / self.weights[0]
            ax.axvline(x=x_boundary, color='k', label='Frontera de Decisión')
        
        ax.set_title(title)
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.legend()
        ax.grid(True)
        
        return fig 