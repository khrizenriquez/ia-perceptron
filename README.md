# Perceptrón Simple - Proyecto de IA 2025

Este proyecto implementa un perceptrón simple para clasificación binaria, desarrollado como parte del Proyecto Final de Inteligencia Artificial (2025).

<img width="1494" alt="image" src="https://github.com/user-attachments/assets/35411d4a-e2fd-4d92-b000-dbdf536b29e2" />

## Características

- Implementación de perceptrón simple para problemas de clasificación binaria
- Interfaz gráfica interactiva para configurar parámetros y visualizar el entrenamiento
- Soporte para problemas predefinidos (AND, OR, NAND) y personalizados
- Visualización de la evolución de pesos y error durante el entrenamiento
- Pruebas unitarias que validan la convergencia en problemas linealmente separables

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
   ```
   git clone https://github.com/usuario/ia-perceptron.git
   cd ia-perceptron
   ```

2. Crear un entorno virtual (opcional pero recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

## Ejecución

### Aplicación web interactiva:

```
streamlit run app.py
```

### Ejecutar pruebas:

```
pytest
```

### Contenedor Docker/Podman:

```
podman build -t io-perceptron .
podman run -d -p 8501:8501 --name io-perceptron io-perceptron
```

## Demo

Una demostración del proyecto está disponible en YouTube: [<<<AQUÍ_URL_VIDEO>>>](<<<AQUÍ_URL_VIDEO>>>)

## Licencia

Este proyecto está bajo la Licencia MIT. 