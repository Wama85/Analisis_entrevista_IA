#!/usr/bin/env python3
"""
Script para ejecutar la API de detección en vivo
"""

import sys
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from live_api import run_api

if __name__ == "__main__":
    run_api()