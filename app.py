"""
Streamlit application for interactive demonstration of the simple perceptron.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from app.perceptron import Perceptron
from app.config import (
    DEFAULT_N_FEATURES, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS,
    DEFAULT_WEIGHT_VALUE, DEFAULT_BIAS_VALUE,
    MIN_FEATURES, MAX_FEATURES, MIN_LEARNING_RATE, MAX_LEARNING_RATE,
    MIN_EPOCHS, MAX_EPOCHS, STEP_EPOCHS, STOP_ON_ALL_CORRECT
)

# Page configuration
st.set_page_config(
    page_title="Perceptrón Simple - IA",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for neon effect and responsive design
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

.stButton>button {
    width: 100%;
}

/* Responsive styling for perceptron visualization */
@media (min-width: 1200px) {
    /* Desktop */
    .perceptron-model-container {
        width: 40%;
        margin: 0 auto;
    }
}

@media (min-width: 768px) and (max-width: 1199px) {
    /* Tablet */
    .perceptron-model-container {
        width: 60%;
        margin: 0 auto;
    }
}

@media (max-width: 767px) {
    /* Mobile */
    .perceptron-model-container {
        width: 90%;
        margin: 0 auto;
    }
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Perceptrón Simple - Demostración Interactiva")
st.markdown("""
Esta aplicación demuestra el funcionamiento de un perceptrón simple en la operación lógica OR.
Puedes configurar los parámetros iniciales y visualizar el proceso de entrenamiento.
""")

# Function to generate dynamic training data based on number of features
def generate_training_data(n_features):
    """Generate training data for n_features dimensions.
    For features > 2, we extend the OR operation by considering it a 1 if any input is 1.
    """
    # Generate all possible binary combinations for n_features
    import itertools
    X = np.array(list(itertools.product([0, 1], repeat=n_features)))
    
    # For OR operation: output is 1 if any input is 1, otherwise 0
    y = np.array([1 if np.sum(x) > 0 else 0 for x in X])
    
    return X, y

# Function to recalculate perceptron predictions based on current weights and bias
def recalculate_predictions(weights, bias, X, y):
    """Recalculate predictions based on given weights and bias."""
    # Create a perceptron with the specified weights and bias (no training)
    perceptron = Perceptron(
        input_size=len(weights),
        weights=np.array(weights),
        bias=bias,
    )
    
    # Append the current weights to the history (to display in the table)
    if not hasattr(perceptron, 'weights_history'):
        perceptron.weights_history = [np.array(weights)]
    
    # Calculate predictions
    predictions = perceptron.predict(X)
    
    # Calculate error rate
    error_rate = np.mean(predictions != y)
    
    return perceptron, predictions, error_rate

# Clear session state when number of features changes
def reset_session_state_for_features(n_features):
    """Reset session state values when number of features changes"""
    # Generate new random weights and bias
    st.session_state.random_weights = [DEFAULT_WEIGHT_VALUE] * n_features
    st.session_state.random_bias = DEFAULT_BIAS_VALUE
    
    # Clear iteration details
    if 'iteration_details' in st.session_state:
        del st.session_state.iteration_details

# Sidebar for configuration
st.sidebar.header("Configuración del Perceptrón")

# Number of features/weights selection
st.sidebar.subheader("Estructura del Perceptrón")

# Function that will be called when n_features changes
def n_features_changed():
    if 'current_n_features' in st.session_state and st.session_state.current_n_features != st.session_state.n_features:
        reset_session_state_for_features(st.session_state.n_features)

# Number input for features with on_change callback
n_features = st.sidebar.number_input(
    "Número de entradas/pesos",
    min_value=MIN_FEATURES,
    max_value=MAX_FEATURES,
    value=DEFAULT_N_FEATURES,
    step=1,
    key="n_features",
    on_change=n_features_changed
)

# Generate or update training data based on number of features
if 'current_n_features' not in st.session_state or st.session_state.current_n_features != n_features:
    X, y = generate_training_data(n_features)
    st.session_state.current_n_features = n_features
    st.session_state.X = X
    st.session_state.y = y
else:
    X = st.session_state.X
    y = st.session_state.y

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
    # Add anchor for scrolling
    st.markdown('<div id="evolucion-de-pesos"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolución de Pesos")
        weights_df = pd.DataFrame(perceptron.weights_history)
        weights_df.columns = [f"w{i+1}" for i in range(weights_df.shape[1])]
        
        # Ensure that bias list length matches the index length
        # The error occurs because the perceptron saves an initial bias value plus values for each epoch
        # but epochs only counts training epochs, not including initialization
        bias_values = perceptron.bias_history
        if len(bias_values) < len(weights_df):
            # If bias values are missing, repeat the last value
            bias_values = bias_values + [bias_values[-1]] * (len(weights_df) - len(bias_values))
        elif len(bias_values) > len(weights_df):
            # If there are too many bias values, trim
            bias_values = bias_values[:len(weights_df)]
        
        weights_df["bias"] = bias_values
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

    # Initialize iteration details in session state if not present
    if 'iteration_details' not in st.session_state:
        # Create iteration data
        st.session_state.iteration_details = []
        for i in range(len(perceptron.weights_history)):
            epoch_data = {
                "Época": i,
            }
            
            # Add weights columns
            for j in range(n_features):
                epoch_data[f"w{j+1}"] = float(perceptron.weights_history[i][j])
            
            epoch_data["bias"] = float(perceptron.bias) if i == 0 else float(perceptron.bias)
            epoch_data["Error"] = float(error_history[i-1]) if i > 0 else None
            
            # Add predictions for each data point
            if i > 0:  # Only for epochs after initialization
                weights = perceptron.weights_history[i]
                bias = perceptron.bias
                
                # Calculate predictions for each point (limit to first 4 points to avoid overcrowding)
                for j, (x_point, y_true) in enumerate(zip(X[:min(len(X), 4)], y[:min(len(y), 4)])):
                    activation = np.dot(x_point, weights) + bias
                    prediction = 1 if activation >= 0 else 0
                    is_correct = prediction == y_true
                    
                    # Format the input point as a string
                    x_str = ",".join([str(int(x)) for x in x_point])
                    epoch_data[f"Entrada ({x_str})"] = f"{'✓' if is_correct else '✗'} (act={activation:.2f})"
            
            st.session_state.iteration_details.append(epoch_data)
    
    # Show iteration details with editing capability
    st.markdown('<div id="detalles-de-las-iteraciones"></div>', unsafe_allow_html=True)
    st.subheader("Detalles de las Iteraciones")
    
    # Key for edited data to persist between reruns
    if 'edited_data_key' not in st.session_state:
        st.session_state.edited_data_key = "edited_iterations_" + str(hash(str(st.session_state.iteration_details)))
    
    # Create an editable dataframe from the iteration details
    edited_iterations = st.data_editor(
        st.session_state.iteration_details,
        disabled=["Época"],  # Prevent editing the epoch column
        hide_index=True,
        key=st.session_state.edited_data_key,
        use_container_width=True,
        on_change=lambda: None,  # Empty callback to prevent page refresh
    )
    
    # Store any user edits back to session state
    if st.session_state.edited_data_key in st.session_state:
        st.session_state.iteration_details = st.session_state[st.session_state.edited_data_key]
    
    # Show predictions
    st.subheader("Predicciones Finales")
    predictions = perceptron.predict(X)
    
    # Create prediction dataframe with sample data (limit rows if too many)
    max_rows_to_show = 8  # Limit the number of rows to show in the UI
    if len(X) > max_rows_to_show:
        X_sample = X[:max_rows_to_show]
        y_sample = y[:max_rows_to_show]
        predictions_sample = predictions[:max_rows_to_show]
        note = f"Mostrando las primeras {max_rows_to_show} predicciones de {len(X)} totales."
    else:
        X_sample = X
        y_sample = y
        predictions_sample = predictions
        note = ""
    
    pred_df = pd.DataFrame({
        "Esperado": y_sample,
        "Predicción": predictions_sample,
        "¿Correcto?": y_sample == predictions_sample
    })
    
    # Add input features as columns
    for i in range(n_features):
        pred_df.insert(i, f"X{i+1}", X_sample[:, i])
    
    st.dataframe(pred_df)
    if note:
        st.info(note)
    
    # New section to visualize predictions by iteration along with iteration details
    st.markdown('<div id="predicciones-detalles-iteracion"></div>', unsafe_allow_html=True)
    st.subheader("Visualización Detallada por Iteración")
    st.markdown("""
    Esta sección muestra la evolución de las predicciones y los detalles técnicos (pesos, bias, error) para cada iteración.
    """)
    
    # Crear un contenedor para mostrar los detalles iteración por iteración
    for iter_idx in range(1, len(perceptron.weights_history)):
        # Crear un expander para cada iteración
        with st.expander(f"Iteración {iter_idx}", expanded=iter_idx == len(perceptron.weights_history)-1):
            # Crear dos columnas para mostrar predicciones y detalles lado a lado
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Predicciones")
                
                # Calcular las predicciones para esta iteración
                weights = perceptron.weights_history[iter_idx]
                bias = perceptron.bias_history[iter_idx]
                
                # Crear un DataFrame para las predicciones de esta iteración
                iter_preds = []
                for j, (x_point, y_true) in enumerate(zip(X_sample, y_sample)):
                    activation = np.dot(x_point, weights) + bias
                    prediction = 1 if activation >= 0 else 0
                    is_correct = prediction == y_true
                    
                    row_data = {"Esperado": y_true, "Predicción": prediction, "Correcto": is_correct}
                    for k in range(n_features):
                        row_data[f"X{k+1}"] = x_point[k]
                    
                    iter_preds.append(row_data)
                
                iter_df = pd.DataFrame(iter_preds)
                
                # Estilizar el DataFrame para resaltar resultados
                def highlight_correct(val):
                    if val is True:
                        return 'background-color: #d4edda'
                    elif val is False:
                        return 'background-color: #f8d7da'
                    return ''
                
                styled_df = iter_df.style.applymap(highlight_correct, subset=['Correcto'])
                
                # Mostrar la tabla de predicciones
                st.dataframe(styled_df, use_container_width=True)
                
                # Mostrar información de error para esta iteración
                error_value = perceptron.error_history[iter_idx-1] if iter_idx-1 < len(perceptron.error_history) else None
                if error_value is not None:
                    st.metric("Tasa de Error", f"{error_value:.4f}")
            
            with col2:
                st.markdown("##### Detalles Técnicos")
                
                # Crear un DataFrame con los detalles de esta iteración
                details_data = {
                    "Parámetro": ["Época"] + [f"Peso {i+1}" for i in range(n_features)] + ["Bias"]
                }
                
                # Agregar los valores
                values = [iter_idx]
                for w in weights:
                    values.append(f"{w:.4f}")
                values.append(f"{bias:.4f}")
                
                details_data["Valor"] = values
                
                # Si no es la primera iteración, agregar cambios respecto a la iteración anterior
                if iter_idx > 1:
                    prev_weights = perceptron.weights_history[iter_idx-1]
                    prev_bias = perceptron.bias_history[iter_idx-1]
                    
                    changes = ["-"]  # No hay cambio para la época
                    for i, (curr, prev) in enumerate(zip(weights, prev_weights)):
                        change = curr - prev
                        changes.append(f"{change:+.4f}")
                    
                    bias_change = bias - prev_bias
                    changes.append(f"{bias_change:+.4f}")
                    
                    details_data["Cambio"] = changes
                
                # Mostrar la tabla de detalles
                details_df = pd.DataFrame(details_data)
                st.dataframe(details_df, use_container_width=True)
                
                # Mostrar información sobre la convergencia
                if iter_idx == len(perceptron.weights_history) - 1:
                    all_correct = all(pred == true for pred, true in zip(perceptron.predict(X), y))
                    if all_correct:
                        st.success("¡Convergencia alcanzada! Todas las predicciones son correctas.")
                    else:
                        incorrect_count = sum(pred != true for pred, true in zip(perceptron.predict(X), y))
                        st.warning(f"No se alcanzó convergencia completa. {incorrect_count} predicciones incorrectas.")
                        
                # Mostrar fórmula del perceptrón para esta iteración
                formula = f"y = 1 si ("
                for i, w in enumerate(weights):
                    if i > 0:
                        formula += " + " if w >= 0 else " - "
                    else:
                        formula += "-" if w < 0 else ""
                    formula += f"{abs(w):.4f} × X{i+1}"
                formula += f" + {bias:.4f}" if bias >= 0 else f" - {abs(bias):.4f}"
                formula += ") ≥ 0, sino y = 0"
                
                st.markdown("##### Fórmula del Perceptrón")
                st.code(formula)

    # Add perceptron graphical visualization
    st.subheader("Visualización Gráfica del Perceptrón")
    
    # Create a figure for the perceptron visualization with a style similar to the reference diagram
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # General configuration
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    
    # Define locations for the components
    box_width = 3
    box_height = 1
    margin = 0.5
    
    # Box positions
    x_center = 5
    y_input1 = 10
    y_input2 = 10
    y_bias = 10
    x_input1 = 2
    x_input2 = 5
    x_bias = 8
    
    y_summation = 7
    y_activation = 4
    y_output = 1
    
    # Draw input boxes
    input1_rect = plt.Rectangle((x_input1 - box_width/2, y_input1 - box_height/2), box_width, box_height, 
                               fill=True, color='white', edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(input1_rect)
    ax.text(x_input1, y_input1, f"Entrada X₁", ha='center', va='center', fontsize=12)
    
    input2_rect = plt.Rectangle((x_input2 - box_width/2, y_input2 - box_height/2), box_width, box_height, 
                               fill=True, color='white', edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(input2_rect)
    ax.text(x_input2, y_input2, f"Entrada X₂", ha='center', va='center', fontsize=12)
    
    bias_rect = plt.Rectangle((x_bias - box_width/2, y_bias - box_height/2), box_width, box_height, 
                             fill=True, color='white', edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(bias_rect)
    ax.text(x_bias, y_bias, f"Umbral b", ha='center', va='center', fontsize=12)
    
    # Draw summation box
    sum_rect = plt.Rectangle((x_center - box_width/2, y_summation - box_height/2), box_width, box_height, 
                           fill=True, color='white', edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(sum_rect)
    ax.text(x_center, y_summation + 0.2, " ", ha='center', va='center', fontsize=12)
    formula = r"$\Sigma(w_ix_i) + b$"
    ax.text(x_center, y_summation - 0.2, formula, ha='center', va='center', fontsize=10)
    
    # Draw activation function box
    act_rect = plt.Rectangle((x_center - box_width/2, y_activation - box_height/2), box_width, box_height, 
                           fill=True, color='white', edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(act_rect)
    ax.text(x_center, y_activation + 0.2, "Función de", ha='center', va='center', fontsize=12)
    ax.text(x_center, y_activation - 0.2, "Activación", ha='center', va='center', fontsize=12)
    
    # Draw output box
    out_rect = plt.Rectangle((x_center - box_width/2, y_output - box_height/2), box_width, box_height, 
                           fill=True, color='white', edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(out_rect)
    ax.text(x_center, y_output + 0.2, "Salida", ha='center', va='center', fontsize=12)
    ax.text(x_center, y_output - 0.2, "(0 o 1)", ha='center', va='center', fontsize=12)
    
    # Connect components with arrows
    arrow_props = dict(arrowstyle='->', linewidth=1.5, color='gray')
    
    # Connect inputs to summation
    ax.annotate("", xy=(x_center, y_summation - box_height/2 - margin/4), 
               xytext=(x_input1, y_input1 + box_height/2 + margin/4),
               arrowprops=arrow_props)
    ax.text((x_center + x_input1)/2 - 0.5, (y_summation + y_input1)/2, f"w₁={perceptron.weights[0]:.2f}", 
           fontsize=10, ha='center', va='center')
    
    ax.annotate("", xy=(x_center, y_summation - box_height/2 - margin/4), 
               xytext=(x_input2, y_input2 + box_height/2 + margin/4),
               arrowprops=arrow_props)
    ax.text((x_center + x_input2)/2, (y_summation + y_input2)/2, f"w₂={perceptron.weights[1]:.2f}", 
           fontsize=10, ha='center', va='center')
    
    ax.annotate("", xy=(x_center, y_summation - box_height/2 - margin/4), 
               xytext=(x_bias, y_bias + box_height/2 + margin/4),
               arrowprops=arrow_props)
    ax.text((x_center + x_bias)/2 + 0.5, (y_summation + y_bias)/2, f"b={perceptron.bias:.2f}", 
           fontsize=10, ha='center', va='center')
    
    # Connect summation to activation function
    ax.annotate("", xy=(x_center, y_activation - box_height/2 - margin/4), 
               xytext=(x_center, y_summation + box_height/2 + margin/4),
               arrowprops=arrow_props)
    
    # Connect activation function to output
    ax.annotate("", xy=(x_center, y_output - box_height/2 - margin/4), 
               xytext=(x_center, y_activation + box_height/2 + margin/4),
               arrowprops=arrow_props)
    
    # Add perceptron model title
    ax.set_title("Modelo del Perceptrón", fontsize=16, pad=20)
    
    # Show the final perceptron formula below the diagram
    formula_text = f"y = 1 si ({perceptron.weights[0]:.2f}·x₁ + {perceptron.weights[1]:.2f}·x₂ + {perceptron.bias:.2f}) ≥ 0, sino 0"
    fig.text(0.5, 0.05, formula_text, ha='center', fontsize=12, 
            bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Convert the matplotlib figure to an image for HTML embedding with responsive class
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    
    # Display the figure using HTML for responsiveness
    st.markdown(f'<div class="perceptron-model-container"><img src="data:image/png;base64,{img_str}" style="width:100%;"></div>', unsafe_allow_html=True)
    plt.close(fig)  # Close the figure to free memory

# Parameter configuration
st.sidebar.subheader("Parámetros")

# Initialize session variables for weights and bias if they don't exist or if number of features has changed
if 'random_weights' not in st.session_state or len(st.session_state.random_weights) != n_features:
    st.session_state.random_weights = [DEFAULT_WEIGHT_VALUE] * n_features
if 'random_bias' not in st.session_state:
    st.session_state.random_bias = DEFAULT_BIAS_VALUE

# Function to randomize weights
def randomize_weights():
    st.session_state.random_weights = np.random.randn(n_features).tolist()
    st.session_state.random_bias = float(np.random.randn())
    # Clear the URL fragment when randomizing weights
    st.query_params.clear()

# Button to randomize weights - Placed before the widgets that use these values
if st.sidebar.button("Aleatorizar Pesos", on_click=randomize_weights):
    st.sidebar.success("¡Pesos inicializados aleatoriamente!")
    st.rerun()

# Initial weights (dynamic based on n_features)
st.sidebar.markdown("**Pesos iniciales**")
weights = []
for i in range(n_features):
    weights.append(st.sidebar.number_input(
        f"Peso w{i+1}", 
        value=st.session_state.random_weights[i] if i < len(st.session_state.random_weights) else DEFAULT_WEIGHT_VALUE, 
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

# Learning rate
learning_rate = st.sidebar.slider(
    "Tasa de aprendizaje (η)", 
    min_value=MIN_LEARNING_RATE, 
    max_value=MAX_LEARNING_RATE, 
    value=DEFAULT_LEARNING_RATE, 
    step=0.01
)

# Maximum epochs
max_epochs = st.sidebar.slider(
    "Máximo de épocas", 
    min_value=MIN_EPOCHS, 
    max_value=MAX_EPOCHS, 
    value=DEFAULT_MAX_EPOCHS, 
    step=STEP_EPOCHS
)

# Opción para configurar el criterio de parada
stop_criterion = st.sidebar.checkbox(
    "Detener cuando todas las predicciones sean correctas", 
    value=STOP_ON_ALL_CORRECT,
    help="Si está marcado, el entrenamiento se detiene cuando todas las predicciones son correctas. "
         "Si no está marcado, se detiene cuando la tasa de error es 0."
)

# Clear iteration details when training parameters change
def clear_iteration_details():
    if 'iteration_details' in st.session_state:
        del st.session_state.iteration_details

# Function to handle training
def train_perceptron():
    clear_iteration_details()
    # Set URL fragment to focus on weights section after training
    st.query_params.clear()
    
    # Update the stopping criterion in the configuration
    global STOP_ON_ALL_CORRECT
    STOP_ON_ALL_CORRECT = stop_criterion

# Button to train the perceptron
if st.sidebar.button("Entrenar Perceptrón", on_click=train_perceptron):
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

# Additional information
st.sidebar.markdown("---")
st.sidebar.info("""
**Información:**
- Para más de 2 entradas, la operación OR se extiende (1 si al menos una entrada es 1).
- En la sección "Visualización de Predicciones por Iteración" puedes ver cómo evoluciona el aprendizaje en cada época.
""")
