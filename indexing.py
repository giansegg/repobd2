"""
FASE 3: Construcción del Índice Invertido Flat
Crea la estructura de índice invertido para búsqueda eficiente
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from utils import (
    save_pickle, load_pickle,
    Timer, PROCESSED_DIR, INDEX_DIR
)


class InvertedIndex:
    """Índice Invertido Flat para búsqueda de audio"""
    
    def __init__(self):
        self.index = {}  # {word_id: [(audio_id, tfidf_score), ...]}
        self.audio_norms = {}  # {audio_id: norm} para similitud coseno
        self.n_words = 0
        self.n_audios = 0
    
    def build_index(
        self, 
        tfidf_matrix: np.ndarray,
        min_score: float = 0.0
    ) -> None:
        """
        Construye el índice invertido desde la matriz TF-IDF
        
        Estructura objetivo:
        {
            0: [(audio_5, 0.23), (audio_12, 0.45), ...],  # palabra 0
            1: [(audio_3, 0.87), (audio_8, 0.12), ...],   # palabra 1
            ...
        }
        
        Args:
            tfidf_matrix: Matriz TF-IDF (n_audios, n_words)
            min_score: Score mínimo para incluir en el índice
        """
        print(f"\n🔨 Construyendo índice invertido")
        print(f"📊 Matriz TF-IDF: {tfidf_matrix.shape}")
        
        self.n_audios, self.n_words = tfidf_matrix.shape
        self.index = defaultdict(list)
        
        with Timer("Construcción del índice"):
            # Por cada palabra acústica (columna)
            for word_id in range(self.n_words):
                # Obtener todos los audios que contienen esta palabra
                for audio_id in range(self.n_audios):
                    score = tfidf_matrix[audio_id, word_id]
                    
                    # Solo guardar si el score es significativo
                    if score > min_score:
                        self.index[word_id].append((audio_id, score))
                
                # Ordenar por score descendente (opcional, para eficiencia)
                if len(self.index[word_id]) > 0:
                    self.index[word_id].sort(key=lambda x: x[1], reverse=True)
        
        # Calcular normas de los documentos (para similitud coseno)
        print(f"📐 Calculando normas de documentos")
        for audio_id in range(self.n_audios):
            norm = np.linalg.norm(tfidf_matrix[audio_id])
            self.audio_norms[audio_id] = norm
        
        # Convertir defaultdict a dict normal
        self.index = dict(self.index)
        
        # Estadísticas
        non_empty_words = len(self.index)
        avg_postings = np.mean([len(postings) for postings in self.index.values()])
        
        print(f"✅ Índice construido:")
        print(f"   - Total palabras: {self.n_words}")
        print(f"   - Palabras con postings: {non_empty_words}")
        print(f"   - Promedio postings/palabra: {avg_postings:.1f}")
        print(f"   - Total audios: {self.n_audios}")
    
    def get_candidates(self, query_vector: np.ndarray) -> set:
        """
        Obtiene audios candidatos usando el índice invertido
        
        Args:
            query_vector: Vector TF-IDF del query
            
        Returns:
            Set de audio_ids candidatos
        """
        candidates = set()
        
        # Solo mirar palabras que aparecen en el query
        for word_id in range(len(query_vector)):
            if query_vector[word_id] > 0:
                if word_id in self.index:
                    # Agregar todos los audios que contienen esta palabra
                    for audio_id, _ in self.index[word_id]:
                        candidates.add(audio_id)
        
        return candidates
    
    def save(self, filepath: str) -> None:
        """Guarda el índice"""
        index_data = {
            'index': self.index,
            'audio_norms': self.audio_norms,
            'n_words': self.n_words,
            'n_audios': self.n_audios
        }
        save_pickle(index_data, filepath)
    
    def load(self, filepath: str) -> None:
        """Carga el índice"""
        index_data = load_pickle(filepath)
        self.index = index_data['index']
        self.audio_norms = index_data['audio_norms']
        self.n_words = index_data['n_words']
        self.n_audios = index_data['n_audios']
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del índice"""
        postings_per_word = [len(postings) for postings in self.index.values()]
        
        return {
            'n_words': self.n_words,
            'n_audios': self.n_audios,
            'non_empty_words': len(self.index),
            'total_postings': sum(postings_per_word),
            'avg_postings': np.mean(postings_per_word),
            'max_postings': max(postings_per_word) if postings_per_word else 0,
            'min_postings': min(postings_per_word) if postings_per_word else 0
        }


def main():
    """Función principal - FASE 3"""
    
    print("="*60)
    print("🎵 FASE 3: CONSTRUCCIÓN DEL ÍNDICE INVERTIDO")
    print("="*60)
    
    # 1. Cargar matriz TF-IDF de la FASE 2
    print("\n📂 Cargando datos de la FASE 2...")
    tfidf_matrix = np.load(PROCESSED_DIR / "tfidf_matrix.npy")
    
    print(f"✅ Matriz TF-IDF cargada: {tfidf_matrix.shape}")
    print(f"   {tfidf_matrix.shape[0]} audios × {tfidf_matrix.shape[1]} palabras acústicas")
    
    # 2. Construir índice invertido
    inverted_index = InvertedIndex()
    inverted_index.build_index(
        tfidf_matrix=tfidf_matrix,
        min_score=0.0  # Incluir todos los scores positivos
    )
    
    # 3. Mostrar estadísticas
    stats = inverted_index.get_stats()
    print(f"\n📊 Estadísticas del Índice:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.2f}")
        else:
            print(f"   - {key}: {value}")
    
    # 4. Guardar índice
    index_path = INDEX_DIR / "inverted_index.pkl"
    inverted_index.save(index_path)
    
    # 5. Prueba rápida del índice
    print(f"\n🧪 Prueba del índice:")
    query_vector = tfidf_matrix[0]  # Usar primer audio como query
    candidates = inverted_index.get_candidates(query_vector)
    print(f"   Query: Audio 0")
    print(f"   Candidatos encontrados: {len(candidates)}")
    print(f"   Reducción: {tfidf_matrix.shape[0]} → {len(candidates)} audios")
    
    # Mostrar ejemplo de estructura del índice
    print(f"\n🔍 Ejemplo de estructura del índice:")
    print(f"   Palabra acústica 0 aparece en {len(inverted_index.index.get(0, []))} audios")
    if 0 in inverted_index.index and len(inverted_index.index[0]) > 0:
        print(f"   Primeros 3 postings:")
        for audio_id, score in inverted_index.index[0][:3]:
            print(f"      - Audio {audio_id}: score = {score:.4f}")
    
    print("\n" + "="*60)
    print("✅ FASE 3 COMPLETADA")
    print("="*60)
    print(f"📁 Archivo generado:")
    print(f"   - {index_path}")
    print(f"\n🎯 Siguiente: Búsqueda KNN (FASE 4)")


if __name__ == "__main__":
    main()