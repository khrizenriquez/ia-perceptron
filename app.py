"""
Streamlit application for interactive demonstration of the simple perceptron.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.perceptron import Perceptron

# Page configuration
st.set_page_config(
    page_title="Perceptrón Simple - IA",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for neon effect
st.markdown("""
<style>
.neon-text {
    color: #fff;
    text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #0073e6, 0 0 20px #0073e6, 0 0 25px #0073e6;
    font-weight: bold;
    font-size: 1.5em;
    animation: neon-glow 1.5s ease-in-out infinite alternate;
}

@keyframes neon-glow {
    from {
        text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #0073e6, 0 0 20px #0073e6;
    }
    to {
        text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #0073e6, 0 0 40px #0073e6;
    }
}

.detailed-table {
    font-size: 0.9em;
}

.highlight-row {
    background-color: #f0f8ff;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Perceptrón Simple - Demostración Interactiva")
st.markdown("""
Esta aplicación demuestra el funcionamiento de un perceptrón simple en la operación lógica OR.
Puedes configurar los parámetros iniciales y visualizar el proceso de entrenamiento.
""")

# OR problem definition
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Show information about OR operation with neon effect
st.markdown('<div class="neon-text">Operación Lógica OR</div>', unsafe_allow_html=True)
st.info("Es 1 cuando al menos una de las entradas es 1.")

# Function to display data tables
def show_data_table(X, y):
    data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
    data["y"] = y
    st.write("Datos de entrenamiento:")
    st.dataframe(data)

# Display training data
show_data_table(X, y)

# Function to display training results
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
        ax.plot(range(len(error_history)), error_history, marker='o', color='blue', linewidth=2)
        ax.set_xlabel("Época")
        ax.set_ylabel("Tasa de Error")
        ax.set_title("Evolución del Error Durante el Entrenamiento")
        ax.grid(True)
        # Add horizontal lines for better visualization
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1, color='red', linestyle=':', alpha=0.5)
        # Set y-axis limits
        ax.set_ylim(-0.05, 1.05)
        st.pyplot(fig)
    
    # Display decision boundary if we have 2 dimensions
    if perceptron.weights.shape[0] == 2:
        st.subheader("Frontera de Decisión")
        decision_boundary = perceptron.plot_decision_boundary(
            X, y, title="Frontera de Decisión - Operación OR"
        )
        st.pyplot(decision_boundary)
    
    # Show iteration details
    st.subheader("Detalles de las Iteraciones")
    
    # Create DataFrame with all details
    iterations_data = []
    for i in range(len(perceptron.weights_history)):
        epoch_data = {
            "Época": i,
            "w1": perceptron.weights_history[i][0],
            "w2": perceptron.weights_history[i][1],
            "bias": perceptron.bias if i == 0 else perceptron.bias,
            "Error": error_history[i-1] if i > 0 else None
        }
        
        # Add predictions for each data point
        if i > 0:  # Only for epochs after initialization
            weights = perceptron.weights_history[i]
            bias = perceptron.bias
            
            # Calculate predictions for each point
            for j, (x_point, y_true) in enumerate(zip(X, y)):
                activation = np.dot(x_point, weights) + bias
                prediction = 1 if activation >= 0 else 0
                is_correct = prediction == y_true
                
                epoch_data[f"Entrada ({x_point[0]},{x_point[1]})"] = f"{'✓' if is_correct else '✗'} (act={activation:.2f})"
        
        iterations_data.append(epoch_data)
    
    # Convert to DataFrame and display
    iterations_df = pd.DataFrame(iterations_data)
    st.dataframe(iterations_df, use_container_width=True)

# Sidebar for configuration
st.sidebar.header("Configuración del Perceptrón")

# Parameter configuration
st.sidebar.subheader("Parámetros")

# Number of features
n_features = X.shape[1]

# Initialize session variables for weights and bias if they don't exist
if 'random_weights' not in st.session_state:
    st.session_state.random_weights = [0.0] * n_features
if 'random_bias' not in st.session_state:
    st.session_state.random_bias = 0.0

# Button to randomize weights - Placed before the widgets that use these values
if st.sidebar.button("Aleatorizar Pesos"):
    # Generate new random weights
    st.session_state.random_weights = np.random.randn(n_features).tolist()
    st.session_state.random_bias = float(np.random.randn())
    st.sidebar.success("¡Pesos inicializados aleatoriamente!")
    st.rerun()

# Initial weights
st.sidebar.markdown("**Pesos iniciales**")
weights = []
for i in range(n_features):
    weights.append(st.sidebar.number_input(
        f"Peso w{i+1}", 
        value=st.session_state.random_weights[i], 
        step=0.1, 
        format="%.2f", 
        key=f"weight_{i}"
    ))

# Bias (threshold)
bias = st.sidebar.number_input(
    "Bias (umbral)", 
    value=st.session_state.random_bias, 
    step=0.1, 
    format="%.2f", 
    key="bias"
)

# Learning rate (default value: 0.5)
learning_rate = st.sidebar.slider("Tasa de aprendizaje (η)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)

# Maximum epochs
max_epochs = st.sidebar.slider("Máximo de épocas", min_value=10, max_value=1000, value=100, step=10)

# Button to train the perceptron
if st.sidebar.button("Entrenar Perceptrón"):
    # Create and train the perceptron
    perceptron = Perceptron(
        input_size=n_features,
        weights=np.array(weights),
        bias=bias,
        learning_rate=learning_rate,
        max_epochs=max_epochs
    )
    
    with st.spinner("Entrenando perceptrón..."):
        error_history, epochs = perceptron.fit(X, y)
    
    # Show convergence message
    if error_history[-1] == 0:
        st.success(f"¡El perceptrón ha convergido en {epochs} épocas!")
    else:
        st.warning(f"El perceptrón no convergió después de {epochs} épocas. Error final: {error_history[-1]:.4f}")
    
    # Show results
    show_training_results(perceptron, epochs, error_history)
    
    # Show predictions
    st.subheader("Predicciones Finales")
    predictions = perceptron.predict(X)
    pred_df = pd.DataFrame({
        "Esperado": y,
        "Predicción": predictions,
        "¿Correcto?": y == predictions
    })
    st.dataframe(pred_df)

# Additional information
st.sidebar.markdown("---")
st.sidebar.info("""
**Información:**
- El perceptrón se detendrá cuando el error sea 0 o se alcance el máximo de épocas.
""")
