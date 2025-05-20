"""
Módulo que implementa un perceptrón simple para clasificación binaria.
"""
from typing import List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


class Perceptron:
    """
    Implementación de un perceptrón simple para clasificación binaria.
    
    Atributos:
        weights (np.ndarray): Pesos sinápticos del perceptrón.
        bias (float): Umbral de activación (sesgo).
        learning_rate (float): Tasa de aprendizaje para ajuste de pesos.
        max_epochs (int): Número máximo de épocas de entrenamiento.
        weights_history (List): Historial de pesos durante el entrenamiento.
        error_history (List): Historial de errores durante el entrenamiento.
    """
    
    def __init__(self, 
                 input_size: int,
                 weights: Optional[np.ndarray] = None,
                 bias: float = 0.0,
                 learning_rate: float = 0.1,
                 max_epochs: int = 100):
        """
        Inicializa el perceptrón con valores configurables.
        
        Args:
            input_size: Dimensión de los datos de entrada.
            weights: Pesos iniciales (si es None, se generan aleatoriamente).
            bias: Valor del umbral (sesgo).
            learning_rate: Tasa de aprendizaje η.
            max_epochs: Número máximo de épocas de entrenamiento.
        """
        if weights is None:
            self.weights = np.random.randn(input_size)
        else:
            self.weights = np.array(weights)
            
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Historiales para seguimiento del entrenamiento
        self.weights_history = [self.weights.copy()]
        self.error_history = []
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza la predicción para las entradas dadas.
        
        Args:
            X: Matriz de datos de entrada, donde cada fila es una muestra.
        
        Returns:
            Predicciones binarias (0 o 1) para cada muestra.
        """
        # Calcular la suma ponderada de entradas y pesos
        activation = np.dot(X, self.weights) + self.bias
        
        # Aplicar función de activación tipo escalón
        return np.where(activation >= 0, 1, 0)
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena el perceptrón durante una época completa.
        
        Args:
            X: Matriz de datos de entrada.
            y: Vector de etiquetas objetivo (0 o 1).
            
        Returns:
            Error total en la época actual.
        """
        error_count = 0
        
        # Iterar sobre cada muestra de entrenamiento
        for i in range(len(X)):
            # Obtener predicción actual
            prediction = self.predict(X[i].reshape(1, -1))[0]
            
            # Calcular el error
            error = y[i] - prediction
            
            # Actualizar contador de errores
            if error != 0:
                error_count += 1
                
                # Actualizar pesos y bias según la regla del perceptrón
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
        
        # Guardar historial de pesos
        self.weights_history.append(self.weights.copy())
        self.bias_history.append(self.bias)
        
        # Calcular tasa de error (proporción de muestras mal clasificadas)
        error_rate = error_count / len(X)
        self.error_history.append(error_rate)
        
        return error_rate
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[float], int]:
        """
        Entrena el perceptrón hasta convergencia o alcanzar max_epochs.
        
        Args:
            X: Matriz de datos de entrada.
            y: Vector de etiquetas objetivo (0 o 1).
            
        Returns:
            Tupla con (historial de errores, número de épocas realizadas).
        """
        # Reiniciar historiales
        self.weights_history = [self.weights.copy()]
        self.error_history = []
        
        # Entrenamiento por épocas
        epoch = 0
        error_rate = 1.0  # Inicializar con error máximo
        
        while error_rate > 0 and epoch < self.max_epochs:
            error_rate = self.train_epoch(X, y)
            epoch += 1
            
            # Si el error es cero, el perceptrón ha convergido
            if error_rate == 0:
                break
                
        return self.error_history, epoch
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                              title: str = "Frontera de decisión") -> plt.Figure:
        """
        Genera un gráfico con la frontera de decisión del perceptrón (solo para 2D).
        
        Args:
            X: Datos de entrada (solo funciona para datos 2D).
            y: Etiquetas reales.
            title: Título del gráfico.
            
        Returns:
            Figura de matplotlib con la frontera de decisión.
        """
        if X.shape[1] != 2:
            raise ValueError("La visualización de frontera solo funciona con datos 2D")
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Graficar puntos
        ax.scatter(X[y==0, 0], X[y==0, 1], label='Clase 0', marker='o')
        ax.scatter(X[y==1, 0], X[y==1, 1], label='Clase 1', marker='x')
        
        # Calcular frontera de decisión
        if self.weights[1] != 0:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min = (-self.bias - self.weights[0] * x_min) / self.weights[1]
            y_max = (-self.bias - self.weights[0] * x_max) / self.weights[1]
            ax.plot([x_min, x_max], [y_min, y_max], 'k-', label='Frontera de decisión')
        else:
            # Caso de frontera vertical
            x_boundary = -self.bias / self.weights[0]
            ax.axvline(x=x_boundary, color='k', label='Frontera de decisión')
        
        ax.set_title(title)
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.legend()
        ax.grid(True)
        
        return fig 