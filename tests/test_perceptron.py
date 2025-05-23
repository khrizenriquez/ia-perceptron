"""
Pruebas unitarias para el perceptrón simple.
"""
import pytest
import numpy as np
from app.perceptron import Perceptron

class TestPerceptron:
    """Suite de pruebas para validar el funcionamiento del perceptrón."""
    
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
        
        # Verificar convergencia (error final = 0)
        assert error_history[-1] == 0, "El perceptrón no converge en el problema OR"
        
        # Verificar predicciones
        predictions = perceptron.predict(X)
        assert np.array_equal(predictions, y), "Las predicciones no coinciden con las etiquetas esperadas"
    
    def test_or_with_fixed_weights(self):
        """Prueba la operación OR con pesos fijos que deberían funcionar."""
        # Inicializar perceptrón con pesos que deberían clasificar correctamente OR
        perceptron = Perceptron(
            input_size=2,
            weights=np.array([1.0, 1.0]),
            bias=-0.5,  # Con estos valores, simula la operación OR
        )
        
        # Datos para operación lógica OR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        
        # Realizar predicciones
        predictions = perceptron.predict(X)
        
        # Verificar predicciones
        assert np.array_equal(predictions, y), "El perceptrón con pesos fijos no clasifica correctamente OR"
    
    def test_or_with_different_learning_rates(self):
        """Prueba la operación OR con diferentes tasas de aprendizaje."""
        # Datos para operación lógica OR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        
        learning_rates = [0.01, 0.1, 0.5, 1.0]
        
        for lr in learning_rates:
            # Inicializar perceptrón con tasa de aprendizaje específica
            perceptron = Perceptron(
                input_size=2,
                learning_rate=lr,
                max_epochs=1000  # Mayor max_epochs para asegurar convergencia con tasas de aprendizaje pequeñas
            )
            
            # Entrenar el perceptrón
            error_history, epochs = perceptron.fit(X, y)
            
            # Verificar convergencia (error final = 0)
            assert error_history[-1] == 0, f"El perceptrón no converge con tasa de aprendizaje {lr}"
            
            # Verificar predicciones
            predictions = perceptron.predict(X)
            assert np.array_equal(predictions, y), f"Las predicciones no coinciden con tasa de aprendizaje {lr}"
    
    def test_or_with_noise(self):
        """Prueba la operación OR con datos ruidosos."""
        # Crear una versión ligeramente ruidosa de los datos OR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], 
                     [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
        # Etiquetas OR normales más entradas ligeramente ruidosas (debería clasificar correctamente)
        y = np.array([0, 1, 1, 1, 0, 1, 1, 1])
        
        # Inicializar perceptrón
        perceptron = Perceptron(
            input_size=2,
            learning_rate=0.1,
            max_epochs=200
        )
        
        # Entrenar el perceptrón
        error_history, epochs = perceptron.fit(X, y)
        
        # Verificar convergencia (error final = 0)
        assert error_history[-1] == 0, "El perceptrón no converge en datos OR ruidosos"
        
        # Verificar predicciones
        predictions = perceptron.predict(X)
        assert np.array_equal(predictions, y), "Las predicciones no coinciden en datos ruidosos"
    
    def test_or_generalization(self):
        """Prueba la capacidad del perceptrón para generalizar la función OR a nuevas entradas."""
        # Datos de entrenamiento para OR
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 1, 1, 1])
        
        # Inicializar perceptrón con pesos fijos para un comportamiento predecible
        perceptron = Perceptron(
            input_size=2,
            weights=np.array([1.0, 1.0]),
            bias=-0.5,  # Con estos valores, simula la operación OR
            learning_rate=0.1,
            max_epochs=100
        )
        
        # No es necesario entrenar ya que usamos pesos fijos que funcionan para OR
        
        # Datos de prueba con valores no vistos durante el entrenamiento
        X_test = np.array([
            [0.2, 0.2],  # Debería estar cerca de [0,0] pero depende de la frontera de decisión exacta
            [0.1, 0.8],  # Debería estar cerca de [0,1] -> 1
            [0.7, 0.1],  # Debería estar cerca de [1,0] -> 1
            [0.9, 0.9]   # Debería estar cerca de [1,1] -> 1
        ])
        
        # Realizar predicciones
        predictions = perceptron.predict(X_test)
        
        # Con los pesos fijos [1.0, 1.0] y bias -0.5:
        # Para [0.2, 0.2]: 0.2 + 0.2 - 0.5 = -0.1 < 0 -> 0
        # Para [0.1, 0.8]: 0.1 + 0.8 - 0.5 = 0.4 > 0 -> 1
        # Para [0.7, 0.1]: 0.7 + 0.1 - 0.5 = 0.3 > 0 -> 1
        # Para [0.9, 0.9]: 0.9 + 0.9 - 0.5 = 1.3 > 0 -> 1
        y_expected = np.array([0, 1, 1, 1])
        
        # Probar generalización
        assert np.array_equal(predictions, y_expected), "El perceptrón falla al generalizar la función OR"
    
    def test_predict_method(self):
        """Prueba la funcionalidad del método predict."""
        # Inicializar perceptrón con pesos y bias fijos
        perceptron = Perceptron(
            input_size=2,
            weights=np.array([1.0, 1.0]),
            bias=-0.5,  # Con estos valores, simula la operación OR
        )
        
        # Datos para operación lógica OR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([0, 1, 1, 1])
        
        # Realizar predicciones
        predictions = perceptron.predict(X)
        
        # Verificar predicciones
        assert np.array_equal(predictions, expected), "El método predict no funciona correctamente"
    
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