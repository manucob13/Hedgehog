#!/bin/bash

# Este script se ejecuta ANTES de que Streamlit corra tu aplicación.

# 1. Instalar dependencias del sistema necesarias
sudo apt-get update && \
sudo apt-get install -y build-essential libffi-dev python3-dev

# 2. Instalar schwab-py (esto puede ser redundante, pero garantiza que esté)
pip install schwab-py

# 3. Instalar el resto de dependencias
pip install -r requirements.txt
