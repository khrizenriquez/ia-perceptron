"""
Aplicación de Streamlit para demostración interactiva del perceptrón simple.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.perceptron import Perceptron

# Configuración de la página
st.set_page_config(
    page_title="Perceptrón Simple - IA",
    page_icon="🧠",
    layout="wide"
)

# Título y descripción
st.title("Perceptrón Simple - Demostración Interactiva")
st.markdown("""
Esta aplicación demuestra el funcionamiento de un perceptrón simple en la operación lógica OR.
Puedes configurar los parámetros iniciales y visualizar el proceso de entrenamiento.
""")

# Definición del problema OR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Mostrar información sobre la operación OR
st.info("""
### Operación lógica OR
Es 1 cuando al menos una de las entradas es 1.
""")

# Función para mostrar tablas de datos
def show_data_table(X, y):
    data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
    data["y"] = y
    st.write("Datos de entrenamiento:")
    st.dataframe(data)

# Mostrar datos de entrenamiento
show_data_table(X, y)

# Función para mostrar resultados de entrenamiento
def show_training_results(perceptron, epochs, error_history):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolución de Pesos")
        weights_df = pd.DataFrame(perceptron.weights_history)
        weights_df.columns = [f"w{i+1}" for i in range(weights_df.shape[1])]
        weights_df["bias"] = [perceptron.bias] + [perceptron.bias] * epochs
        weights_df.index.name = "Época"
        st.dataframe(weights_df)
    
    with col2:
        st.subheader("Curva de Error")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(error_history)), error_history, marker='o')
        ax.set_xlabel("Época")
        ax.set_ylabel("Tasa de Error")
        ax.set_title("Evolución del Error Durante el Entrenamiento")
        ax.grid(True)
        st.pyplot(fig)
    
    # Mostrar frontera de decisión si tenemos 2 dimensiones
    if perceptron.weights.shape[0] == 2:
        st.subheader("Frontera de Decisión")
        decision_boundary = perceptron.plot_decision_boundary(
            X, y, title="Frontera de decisión - Operación OR"
        )
        st.pyplot(decision_boundary)

# Panel lateral para configuración
st.sidebar.header("Configuración del Perceptrón")

# Configuración de parámetros
st.sidebar.subheader("Parámetros")

# Número de características
n_features = X.shape[1]

# Pesos iniciales
st.sidebar.markdown("**Pesos iniciales**")
weights = []
for i in range(n_features):
    weights.append(st.sidebar.number_input(f"Peso w{i+1}", value=0.0, step=0.1, format="%.2f"))

# Bias (umbral)
bias = st.sidebar.number_input("Bias (umbral)", value=0.0, step=0.1, format="%.2f")

# Tasa de aprendizaje
learning_rate = st.sidebar.slider("Tasa de aprendizaje (η)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# Máximo de épocas
max_epochs = st.sidebar.slider("Máximo de épocas", min_value=10, max_value=1000, value=100, step=10)

# Botón para inicializar pesos aleatorios
if st.sidebar.button("Randomizar Pesos"):
    weights = np.random.randn(n_features)
    bias = np.random.randn()
    st.sidebar.success("¡Pesos inicializados aleatoriamente!")
    st.rerun()

# Botón para entrenar el perceptrón
if st.sidebar.button("Entrenar Perceptrón"):
    # Crear y entrenar el perceptrón
    perceptron = Perceptron(
        input_size=n_features,
        weights=np.array(weights),
        bias=bias,
        learning_rate=learning_rate,
        max_epochs=max_epochs
    )
    
    with st.spinner("Entrenando perceptrón..."):
        error_history, epochs = perceptron.fit(X, y)
    
    # Mostrar mensaje de convergencia
    if error_history[-1] == 0:
        st.success(f"¡El perceptrón ha convergido en {epochs} épocas!")
    else:
        st.warning(f"El perceptrón no convergió después de {epochs} épocas. Error final: {error_history[-1]:.4f}")
    
    # Mostrar resultados
    show_training_results(perceptron, epochs, error_history)
    
    # Mostrar predicciones
    st.subheader("Predicciones finales")
    predictions = perceptron.predict(X)
    pred_df = pd.DataFrame({
        "Esperado": y,
        "Predicción": predictions,
        "¿Correcto?": y == predictions
    })
    st.dataframe(pred_df)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
**Información:**
- El perceptrón se detendrá cuando el error sea 0 o se alcance el máximo de épocas.
""")
