"""
Módulo con utilidades para el proyecto del perceptrón.
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
    Genera un conjunto de datos aleatorio linealmente separable.
    
    Args:
        n_samples: Número de muestras a generar.
        n_features: Número de características para cada muestra.
        noise: Nivel de ruido en los datos.
        
    Returns:
        Tupla (X, y) con datos de entrada y etiquetas.
    """
    # Generar pesos aleatorios para definir el hiperplano separador
    weights = np.random.randn(n_features)
    bias = np.random.randn()
    
    # Generar datos aleatorios
    X = np.random.randn(n_samples, n_features)
    
    # Calcular etiquetas usando el hiperplano definido por weights y bias
    y_clean = np.where(np.dot(X, weights) + bias >= 0, 1, 0)
    
    # Añadir ruido (cambiar algunas etiquetas aleatoriamente)
    noise_mask = np.random.random(n_samples) < noise
    y = np.copy(y_clean)
    y[noise_mask] = 1 - y[noise_mask]  # Invertir etiquetas con ruido
    
    return X, y

def visualize_training_history(
    epochs: List[int],
    error_history: List[float],
    weights_history: List[np.ndarray]
) -> Tuple[Figure, Figure]:
    """
    Genera visualizaciones del historial de entrenamiento.
    
    Args:
        epochs: Lista de épocas.
        error_history: Historial de errores.
        weights_history: Historial de pesos.
        
    Returns:
        Tupla de figuras (error_fig, weights_fig).
    """
    # Figura para la curva de error
    error_fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, error_history, 'b-', marker='o')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Error')
    ax1.set_title('Evolución del Error Durante el Entrenamiento')
    ax1.grid(True)
    
    # Figura para la evolución de pesos
    weights_fig, ax2 = plt.subplots(figsize=(10, 5))
    weights_array = np.array(weights_history)
    
    for i in range(weights_array.shape[1]):
        ax2.plot(epochs, weights_array[:, i], marker='o', label=f'w{i+1}')
    
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Valor del Peso')
    ax2.set_title('Evolución de los Pesos Durante el Entrenamiento')
    ax2.legend()
    ax2.grid(True)
    
    return error_fig, weights_fig

def evaluate_perceptron(
    predictions: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, Any]:
    """
    Evalúa el rendimiento del perceptrón.
    
    Args:
        predictions: Predicciones realizadas por el perceptrón.
        true_labels: Etiquetas verdaderas.
        
    Returns:
        Diccionario con métricas de evaluación.
    """
    # Calcular métricas básicas
    correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = correct / total
    
    # Calcular verdaderos positivos, falsos positivos, etc.
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    # Calcular métricas adicionales
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