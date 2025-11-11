"""
FASE 4: Búsqueda KNN (Secuencial e Indexado)
Implementa ambos métodos de búsqueda y los compara
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict
import time

from utils import (
    load_pickle, cosine_similarity,
    Timer, PROCESSED_DIR, INDEX_DIR
)
from codebook import AcousticCodebook
from indexing import InvertedIndex
from Extraccion import AudioFeatureExtractor


class AudioSearchEngine:
    """Motor de búsqueda de audio por similitud"""
    
    def __init__(self):
        self.codebook = None
        self.inverted_index = None
        self.tfidf_matrix = None
        self.metadata = None
        self.extractor = None
    
    def load_index(self) -> None:
        """Carga todos los componentes necesarios"""
        print("📂 Cargando índice y componentes...")
        
        # Cargar codebook
        self.codebook = AcousticCodebook()
        self.codebook.load(INDEX_DIR / "codebook.pkl")
        print("✅ Codebook cargado")
        
        # Cargar índice invertido
        self.inverted_index = InvertedIndex()
        self.inverted_index.load(INDEX_DIR / "inverted_index.pkl")
        print("✅ Índice invertido cargado")
        
        # Cargar matriz TF-IDF
        self.tfidf_matrix = np.load(PROCESSED_DIR / "tfidf_matrix.npy")
        print(f"✅ Matriz TF-IDF cargada: {self.tfidf_matrix.shape}")
        
        # Cargar metadata
        self.metadata = load_pickle(PROCESSED_DIR / "metadata.pkl")
        print(f"✅ Metadata cargada: {len(self.metadata)} audios")
        
        # Inicializar extractor
        self.extractor = AudioFeatureExtractor()
        print("✅ Extractor inicializado")
    
    def process_query(self, query_audio_path: str) -> np.ndarray:
        """
        Procesa un audio de consulta y lo convierte a vector TF-IDF
        
        Args:
            query_audio_path: Ruta del audio de consulta
            
        Returns:
            Vector TF-IDF del query
        """
        # 1. Extraer descriptores MFCC
        descriptors = self.extractor.extract_mfcc(query_audio_path)
        
        if len(descriptors) == 0:
            raise ValueError("No se pudieron extraer características del audio")
        
        # 2. Convertir a histograma
        histogram = self.codebook.audio_to_histogram(descriptors)
        
        # 3. Aplicar TF-IDF
        query_tfidf = self.codebook.tfidf_transformer.transform([histogram]).toarray()[0]
        
        return query_tfidf
    
    def knn_sequential(
        self, 
        query_vector: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Búsqueda KNN Secuencial (baseline)
        Compara con TODOS los audios del dataset
        
        Args:
            query_vector: Vector TF-IDF del query
            k: Número de resultados a retornar
            
        Returns:
            Lista de (audio_id, similarity) ordenada por similitud
        """
        # Usar heap para mantener top-k eficientemente
        heap = []
        
        # Comparar con todos los audios
        for audio_id in range(self.tfidf_matrix.shape[0]):
            audio_vector = self.tfidf_matrix[audio_id]
            
            # Calcular similitud coseno
            similarity = cosine_similarity(query_vector, audio_vector)
            
            # Mantener top-k usando heap (min-heap negativo)
            if len(heap) < k:
                heapq.heappush(heap, (similarity, audio_id))
            else:
                if similarity > heap[0][0]:
                    heapq.heapreplace(heap, (similarity, audio_id))
        
        # Convertir heap a lista ordenada descendente
        results = [(audio_id, sim) for sim, audio_id in sorted(heap, reverse=True)]
        
        return results
    
    def knn_indexed(
        self, 
        query_vector: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Búsqueda KNN con Índice Invertido (optimizado)
        Solo compara con audios candidatos
        
        Args:
            query_vector: Vector TF-IDF del query
            k: Número de resultados a retornar
            
        Returns:
            Lista de (audio_id, similarity) ordenada por similitud
        """
        # 1. Obtener candidatos usando índice invertido
        candidates = self.inverted_index.get_candidates(query_vector)
        
        if len(candidates) == 0:
            return []
        
        # 2. Calcular similitud solo con candidatos
        heap = []
        
        for audio_id in candidates:
            audio_vector = self.tfidf_matrix[audio_id]
            
            # Calcular similitud coseno
            similarity = cosine_similarity(query_vector, audio_vector)
            
            # Mantener top-k
            if len(heap) < k:
                heapq.heappush(heap, (similarity, audio_id))
            else:
                if similarity > heap[0][0]:
                    heapq.heapreplace(heap, (similarity, audio_id))
        
        # Convertir heap a lista ordenada
        results = [(audio_id, sim) for sim, audio_id in sorted(heap, reverse=True)]
        
        return results
    
    def search(
        self, 
        query_audio_path: str, 
        k: int = 10,
        method: str = "indexed"
    ) -> Tuple[List[Dict], float, int]:
        """
        Búsqueda completa de audio
        
        Args:
            query_audio_path: Ruta del audio de consulta
            k: Top-K resultados
            method: "sequential" o "indexed"
            
        Returns:
            (resultados, tiempo_ejecucion, num_candidatos)
        """
        # Procesar query
        query_vector = self.process_query(query_audio_path)
        
        # Ejecutar búsqueda según método
        start_time = time.time()
        
        if method == "sequential":
            results = self.knn_sequential(query_vector, k)
            num_candidates = self.tfidf_matrix.shape[0]
        else:  # indexed
            results = self.knn_indexed(query_vector, k)
            num_candidates = len(self.inverted_index.get_candidates(query_vector))
        
        elapsed_time = time.time() - start_time
        
        # Enriquecer resultados con metadata
        enriched_results = []
        for audio_id, similarity in results:
            result = {
                'audio_id': audio_id,
                'similarity': similarity,
                'file_path': self.metadata[audio_id]['file_path'],
                'file_name': self.metadata[audio_id]['file_name']
            }
            enriched_results.append(result)
        
        return enriched_results, elapsed_time, num_candidates


def main():
    """Función principal - FASE 4 (Demo)"""
    
    print("="*60)
    print("🎵 FASE 4: BÚSQUEDA KNN")
    print("="*60)
    
    # Inicializar motor de búsqueda
    print("\n🚀 Inicializando motor de búsqueda...")
    search_engine = AudioSearchEngine()
    search_engine.load_index()
    
    # Prueba con el primer audio del dataset
    query_path = search_engine.metadata[0]['file_path']
    print(f"\n🎵 Query de prueba: {query_path}")
    
    K = 10
    
    # Búsqueda secuencial
    print(f"\n🔍 Búsqueda KNN Secuencial (K={K})")
    with Timer("KNN Secuencial"):
        results_seq, time_seq, candidates_seq = search_engine.search(
            query_path, k=K, method="sequential"
        )
    
    print(f"   Candidatos evaluados: {candidates_seq}")
    print(f"   Tiempo: {time_seq:.4f} segundos")
    
    # Búsqueda indexada
    print(f"\n🔍 Búsqueda KNN Indexada (K={K})")
    with Timer("KNN Indexada"):
        results_idx, time_idx, candidates_idx = search_engine.search(
            query_path, k=K, method="indexed"
        )
    
    print(f"   Candidatos evaluados: {candidates_idx}")
    print(f"   Tiempo: {time_idx:.4f} segundos")
    
    # Comparación
    print(f"\n📊 Comparación:")
    if time_seq > 0:
        speedup = time_seq / time_idx if time_idx > 0 else float('inf')
        print(f"   Speedup: {speedup:.2f}x más rápido")
    print(f"   Reducción candidatos: {candidates_seq} → {candidates_idx} ({100*(1-candidates_idx/candidates_seq):.1f}%)")
    
    # Mostrar resultados
    print(f"\n🎵 Top-5 Resultados (Indexado):")
    for i, result in enumerate(results_idx[:5], 1):
        print(f"   {i}. {result['file_name']}")
        print(f"      Similarity: {result['similarity']:.4f}")
    
    print("\n" + "="*60)
    print("✅ FASE 4 COMPLETADA")
    print("="*60)
    print(f"\n💡 El motor de búsqueda está listo para usarse")
    print(f"   Puedes buscar audios similares a cualquier MP3")


if __name__ == "__main__":
    main()