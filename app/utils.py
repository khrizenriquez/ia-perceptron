"""
Utility module for the perceptron project.
"""
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def generate_random_dataset(
    n_samples: int = 100,
    n_features: int = 2,
    noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a linearly separable random dataset.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features for each sample.
        noise: Noise level in the data.
        
    Returns:
        Tuple (X, y) with input data and labels.
    """
    # Generate random weights to define the separator hyperplane
    weights = np.random.randn(n_features)
    bias = np.random.randn()
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    
    # Calculate labels using the hyperplane defined by weights and bias
    y_clean = np.where(np.dot(X, weights) + bias >= 0, 1, 0)
    
    # Add noise (randomly change some labels)
    noise_mask = np.random.random(n_samples) < noise
    y = np.copy(y_clean)
    y[noise_mask] = 1 - y[noise_mask]  # Invert labels with noise
    
    return X, y

def visualize_training_history(
    epochs: List[int],
    error_history: List[float],
    weights_history: List[np.ndarray]
) -> Tuple[Figure, Figure]:
    """
    Generates visualizations of the training history.
    
    Args:
        epochs: List of epochs.
        error_history: Error history.
        weights_history: Weights history.
        
    Returns:
        Tuple of figures (error_fig, weights_fig).
    """
    # Figure for the error curve
    error_fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, error_history, 'b-', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.set_title('Error Evolution During Training')
    ax1.grid(True)
    
    # Figure for weights evolution
    weights_fig, ax2 = plt.subplots(figsize=(10, 5))
    weights_array = np.array(weights_history)
    
    for i in range(weights_array.shape[1]):
        ax2.plot(epochs, weights_array[:, i], marker='o', label=f'w{i+1}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Weight Evolution During Training')
    ax2.legend()
    ax2.grid(True)
    
    return error_fig, weights_fig

def evaluate_perceptron(
    predictions: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates the perceptron performance.
    
    Args:
        predictions: Predictions made by the perceptron.
        true_labels: True labels.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    # Calculate basic metrics
    correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = correct / total
    
    # Calculate true positives, false positives, etc.
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    } 