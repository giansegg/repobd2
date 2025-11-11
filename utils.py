"""
Módulo de utilidades para el proyecto de búsqueda de audio
Contiene funciones auxiliares comunes
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

# Configuración de rutas
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
FMA_DIR = DATA_DIR / "fma_small"

# Crear directorios si no existen
for directory in [DATA_DIR, PROCESSED_DIR, INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, filepath: str) -> None:
    """Guarda un objeto en formato pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Guardado: {filepath}")


def load_pickle(filepath: str) -> Any:
    """Carga un objeto desde pickle"""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"✅ Cargado: {filepath}")
    return obj


def save_json(obj: Any, filepath: str) -> None:
    """Guarda un objeto en formato JSON"""
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"✅ Guardado: {filepath}")


def load_json(filepath: str) -> Any:
    """Carga un objeto desde JSON"""
    with open(filepath, 'r') as f:
        obj = json.load(f)
    print(f"✅ Cargado: {filepath}")
    return obj


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos vectores
    
    Args:
        vec1: Vector 1
        vec2: Vector 2
        
    Returns:
        Similitud coseno (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def get_audio_files(directory: str, extensions: List[str] = ['.mp3', '.wav']) -> List[Path]:
    """
    Obtiene todos los archivos de audio en un directorio
    
    Args:
        directory: Ruta del directorio
        extensions: Extensiones de audio a buscar
        
    Returns:
        Lista de rutas de archivos
    """
    audio_files = []
    directory_path = Path(directory)
    
    for ext in extensions:
        audio_files.extend(directory_path.rglob(f"*{ext}"))
    
    return sorted(audio_files)


class Timer:
    """Contexto para medir tiempo de ejecución"""
    
    def __init__(self, name: str = "Operación"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.3f} segundos")


def print_stats(data: np.ndarray, name: str = "Datos") -> None:
    """Imprime estadísticas de un array"""
    print(f"\n📊 Estadísticas de {name}:")
    print(f"   Shape: {data.shape}")
    print(f"   Min: {data.min():.4f}")
    print(f"   Max: {data.max():.4f}")
    print(f"   Mean: {data.mean():.4f}")
    print(f"   Std: {data.std():.4f}")


def format_time(seconds: float) -> str:
    """Formatea segundos en formato legible"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


if __name__ == "__main__":
    print("🔧 Módulo de utilidades cargado")
    print(f"📁 Directorio del proyecto: {PROJECT_ROOT}")
    print(f"📁 Directorio de datos: {DATA_DIR}")