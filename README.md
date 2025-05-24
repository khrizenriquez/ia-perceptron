# Perceptrón Simple - Proyecto de IA 2025

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/khrizenriquez/ia-perceptron)

Este proyecto implementa un perceptrón simple para clasificación binaria, desarrollado como parte del Proyecto Final de Inteligencia Artificial (2025).

<img width="1510" alt="image" src="https://github.com/user-attachments/assets/255566f2-679e-48b9-bb45-8a1e37ecf359" />

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

## Ejecución con contenedores

### Forma sencilla (script de inicio)

Usa el script `start-perceptron.sh` para construir y ejecutar el contenedor en un solo paso:

```bash
# Dar permisos de ejecución al script (solo la primera vez)
chmod +x start-perceptron.sh

# Ejecutar el script
./start-perceptron.sh
```

### Forma manual

Si prefieres ejecutar los comandos manualmente:

```bash
# Construir imagen
podman build -t io-perceptron .
# o con Docker:
# docker build -t io-perceptron .

# Ejecutar contenedor
podman run -d --name io-perceptron -p 8501:8501 io-perceptron
# o con Docker:
# docker run -d --name io-perceptron -p 8501:8501 io-perceptron
```

Después de cualquiera de estos métodos, la aplicación estará disponible en [http://localhost:8501](http://localhost:8501)

## Guía de uso

1. Configura el número de entradas/pesos (predeterminado: 2)
2. Establece los pesos iniciales, bias y tasa de aprendizaje
3. Opcionalmente, utiliza el botón "Aleatorizar Pesos" para valores aleatorios
4. Haz clic en "Entrenar Perceptrón" para iniciar el entrenamiento
5. Una vez entrenado, puedes:
   - Ver la evolución de los pesos y bias
   - Explorar la curva de error
   - Observar la frontera de decisión (solo para 2 dimensiones)
   - Explorar los detalles de cada iteración en la tabla "Detalles de las Iteraciones"

## Estructura del proyecto

```
ia-perceptron/
├── app/                         # Código fuente de la aplicación
│   ├── __init__.py              # Marca el directorio como paquete Python
│   ├── config.py                # Configuración y valores por defecto
│   └── perceptron.py            # Implementación del perceptrón simple
├── tests/                       # Pruebas unitarias
│   ├── __init__.py              # Marca el directorio como paquete Python
│   └── test_perceptron.py       # Pruebas para validar el perceptrón
├── app.py                       # Interfaz de Streamlit
├── Dockerfile                   # Definición del contenedor
├── requirements.txt             # Dependencias del proyecto
├── start-perceptron.sh          # Script para iniciar el contenedor
└── README.md                    # Este archivo
```

## Demo

Una demostración del proyecto está disponible en YouTube: [<<<AQUÍ_URL_VIDEO>>>](<<<AQUÍ_URL_VIDEO>>>)

La aplicación está disponible en línea a través de Streamlit Cloud: [https://ia-perceptron-gmf6yadczjwzyu3cspvljw.streamlit.app/](https://ia-perceptron-gmf6yadczjwzyu3cspvljw.streamlit.app/)


## Licencia

Este proyecto es de código abierto, disponible para uso educativo y personal. 