"""
FASE 2: Construcción del Codebook (Diccionario Acústico)
Aplica K-Means para crear palabras acústicas y convierte audios a histogramas TF-IDF
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from typing import List
from utils import (
    save_pickle, load_pickle,
    Timer, PROCESSED_DIR, INDEX_DIR
)


class AcousticCodebook:
    """Creador del diccionario de palabras acústicas"""
    
    def __init__(self, n_clusters: int = 500):
        """
        Args:
            n_clusters: Número de palabras acústicas (K)
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.codebook = None  # Centroides (palabras acústicas)
        self.tfidf_transformer = None
    
    def build_codebook(self, descriptors: np.ndarray) -> None:
        """
        Construye el codebook usando K-Means
        
        Args:
            descriptors: Array de descriptores (n_samples, 13)
        """
        print(f"\n🎯 Construyendo codebook con K={self.n_clusters}")
        print(f"📊 Descriptores de entrada: {descriptors.shape}")
        
        with Timer("K-Means clustering"):
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=1
            )
            self.kmeans.fit(descriptors)
        
        # Los centroides son las "acoustic words"
        self.codebook = self.kmeans.cluster_centers_
        
        print(f"✅ Codebook creado: {self.codebook.shape}")
        print(f"   Cada audio será representado por {self.n_clusters} palabras acústicas")
    
    def audio_to_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Convierte descriptores de UN audio a histograma de acoustic words
        
        Args:
            descriptors: Descriptores de un audio (n_frames, 13)
            
        Returns:
            Histograma de frecuencias (n_clusters,)
        """
        if len(descriptors) == 0:
            return np.zeros(self.n_clusters)
        
        # Asignar cada descriptor al codeword más cercano
        labels = self.kmeans.predict(descriptors)
        
        # Crear histograma: contar frecuencia de cada palabra acústica
        histogram = np.zeros(self.n_clusters)
        for label in labels:
            histogram[label] += 1
        
        # Normalizar (TF - Term Frequency)
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def convert_dataset_to_histograms(
        self, 
        all_descriptors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Convierte TODO el dataset a histogramas
        
        Args:
            all_descriptors: Lista de descriptores por audio
            
        Returns:
            Matriz de histogramas (n_audios, n_clusters)
        """
        print(f"\n📊 Convirtiendo {len(all_descriptors)} audios a histogramas")
        
        histograms = []
        for descriptors in tqdm(all_descriptors, desc="Generando histogramas"):
            hist = self.audio_to_histogram(descriptors)
            histograms.append(hist)
        
        histograms = np.array(histograms)
        print(f"✅ Histogramas generados: {histograms.shape}")
        print(f"   Cada fila = 1 audio representado por {self.n_clusters} frecuencias")
        
        return histograms
    
    def apply_tfidf(self, histograms: np.ndarray) -> np.ndarray:
        """
        Aplica TF-IDF a los histogramas
        
        TF (Term Frequency): Ya está en el histograma
        IDF (Inverse Document Frequency): Da más peso a palabras raras
        
        Args:
            histograms: Matriz de histogramas (n_audios, n_clusters)
            
        Returns:
            Matriz TF-IDF (n_audios, n_clusters)
        """
        print(f"\n🔢 Aplicando TF-IDF")
        print(f"   TF: Frecuencia de cada palabra en el audio")
        print(f"   IDF: Importancia de la palabra en toda la colección")
        
        with Timer("TF-IDF transformation"):
            self.tfidf_transformer = TfidfTransformer()
            tfidf_matrix = self.tfidf_transformer.fit_transform(histograms)
            tfidf_dense = tfidf_matrix.toarray()
        
        print(f"✅ TF-IDF aplicado: {tfidf_dense.shape}")
        print(f"   Min: {tfidf_dense.min():.4f}, Max: {tfidf_dense.max():.4f}")
        
        return tfidf_dense
    
    def save(self, filepath: str) -> None:
        """Guarda el codebook completo"""
        codebook_data = {
            'kmeans': self.kmeans,
            'codebook': self.codebook,
            'tfidf_transformer': self.tfidf_transformer,
            'n_clusters': self.n_clusters
        }
        save_pickle(codebook_data, filepath)
    
    def load(self, filepath: str) -> None:
        """Carga el codebook"""
        codebook_data = load_pickle(filepath)
        self.kmeans = codebook_data['kmeans']
        self.codebook = codebook_data['codebook']
        self.tfidf_transformer = codebook_data['tfidf_transformer']
        self.n_clusters = codebook_data['n_clusters']


def main():
    """Función principal - FASE 2"""
    
    print("="*60)
    print("🎵 FASE 2: CONSTRUCCIÓN DEL CODEBOOK")
    print("="*60)
    
    # 1. Cargar descriptores de la FASE 1
    print("\n📂 Cargando datos de la FASE 1...")
    descriptors_flat = np.load(PROCESSED_DIR / "descriptors_flat.npy")
    all_descriptors = load_pickle(PROCESSED_DIR / "all_descriptors.pkl")
    
    print(f"✅ Descriptores cargados: {descriptors_flat.shape}")
    print(f"   Total audios: {len(all_descriptors)}")
    print(f"   Descriptores por audio: ~{len(descriptors_flat) // len(all_descriptors)}")
    
    # 2. Construir codebook con K-Means
    K = 500  # Número de palabras acústicas
    print(f"\n🎯 K = {K} palabras acústicas")
    
    codebook_builder = AcousticCodebook(n_clusters=K)
    codebook_builder.build_codebook(descriptors_flat)
    
    # 3. Convertir audios a histogramas
    histograms = codebook_builder.convert_dataset_to_histograms(all_descriptors)
    
    # Guardar histogramas raw
    histograms_path = PROCESSED_DIR / "histograms.npy"
    np.save(histograms_path, histograms)
    print(f"\n💾 Guardado: {histograms_path}")
    
    # 4. Aplicar TF-IDF
    tfidf_matrix = codebook_builder.apply_tfidf(histograms)
    
    # Guardar TF-IDF
    tfidf_path = PROCESSED_DIR / "tfidf_matrix.npy"
    np.save(tfidf_path, tfidf_matrix)
    print(f"💾 Guardado: {tfidf_path}")
    
    # 5. Guardar codebook completo
    codebook_path = INDEX_DIR / "codebook.pkl"
    codebook_builder.save(codebook_path)
    print(f"💾 Guardado: {codebook_path}")
    
    # Estadísticas finales
    print("\n" + "="*60)
    print("✅ FASE 2 COMPLETADA")
    print("="*60)
    print(f"📁 Archivos generados:")
    print(f"   - Histogramas: {histograms_path}")
    print(f"   - TF-IDF: {tfidf_path}")
    print(f"   - Codebook: {codebook_path}")
    print(f"\n📊 Resumen:")
    print(f"   - Audios procesados: {len(all_descriptors)}")
    print(f"   - Palabras acústicas: {K}")
    print(f"   - Dimensión TF-IDF: {tfidf_matrix.shape}")
    print(f"\n🎯 Siguiente: Construcción del Índice Invertido (FASE 3)")


if __name__ == "__main__":
    main()