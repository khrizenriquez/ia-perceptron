"""
Pruebas unitarias para el perceptrón simple.
"""
import pytest
import numpy as np
from app.perceptron import Perceptron

class TestPerceptron:
    """Suite de pruebas para validar el funcionamiento del perceptrón."""
    
    def test_and_convergence(self):
        """Prueba que el perceptrón converge en el problema AND."""
        # Datos para operación lógica AND
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        
        # Inicializar perceptrón con pesos aleatorios
        perceptron = Perceptron(
            input_size=2,
            learning_rate=0.1,
            max_epochs=100
        )
        
        # Entrenar el perceptrón
        error_history, epochs = perceptron.fit(X, y)
        
        # Verificar que converge (error final = 0)
        assert error_history[-1] == 0, "El perceptrón no converge en el problema AND"
        
        # Verificar predicciones
        predictions = perceptron.predict(X)
        assert np.array_equal(predictions, y), "Las predicciones no coinciden con las etiquetas esperadas"
    
    def test_or_convergence(self):
        """Prueba que el perceptrón converge en el problema OR."""
        # Datos para operación lógica OR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        
        # Inicializar perceptrón con pesos aleatorios
        perceptron = Perceptron(
            input_size=2,
            learning_rate=0.1,
            max_epochs=100
        )
        
        # Entrenar el perceptrón
        error_history, epochs = perceptron.fit(X, y)
        
        # Verificar que converge (error final = 0)
        assert error_history[-1] == 0, "El perceptrón no converge en el problema OR"
        
        # Verificar predicciones
        predictions = perceptron.predict(X)
        assert np.array_equal(predictions, y), "Las predicciones no coinciden con las etiquetas esperadas"
    
    def test_no_convergence_xor(self):
        """Prueba que el perceptrón no converge en el problema XOR (no linealmente separable)."""
        # Datos para operación lógica XOR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        
        # Inicializar perceptrón con pesos aleatorios
        perceptron = Perceptron(
            input_size=2,
            learning_rate=0.1,
            max_epochs=100
        )
        
        # Entrenar el perceptrón
        error_history, epochs = perceptron.fit(X, y)
        
        # Verificar que no converge (error final > 0)
        assert error_history[-1] > 0, "El perceptrón converge en XOR, lo cual no debería ocurrir"
        
        # Verificar que se detiene al alcanzar max_epochs
        assert epochs == perceptron.max_epochs, "El entrenamiento no se detuvo al alcanzar max_epochs"
    
    def test_predict_method(self):
        """Prueba el funcionamiento del método predict."""
        # Inicializar perceptrón con pesos y bias fijos
        perceptron = Perceptron(
            input_size=2,
            weights=np.array([1.0, 1.0]),
            bias=-1.5,  # Con estos valores, simula operación AND
        )
        
        # Datos para operación lógica AND
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([0, 0, 0, 1])
        
        # Realizar predicciones
        predictions = perceptron.predict(X)
        
        # Verificar predicciones
        assert np.array_equal(predictions, expected), "El método predict no funciona correctamente" 