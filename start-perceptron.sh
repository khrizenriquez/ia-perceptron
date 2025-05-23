#!/bin/bash

# Script para construir y ejecutar el contenedor del perceptrón

echo "Deteniendo y eliminando el contenedor anterior (si existe)..."
podman stop io-perceptron 2>/dev/null || true
podman rm io-perceptron 2>/dev/null || true

echo "Construyendo la imagen del contenedor..."
podman build -t io-perceptron .

echo "Iniciando el contenedor en modo detached..."
podman run -d --name io-perceptron -p 8501:8501 io-perceptron

echo "¡Listo! La aplicación está disponible en http://localhost:8501" 