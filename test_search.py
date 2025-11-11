"""
Script de Prueba Simple
Permite hacer búsquedas de audio de forma interactiva
"""

from search import AudioSearchEngine
from pathlib import Path

def test_search():
    """Prueba el motor de búsqueda con un audio"""
    
    print("="*60)
    print("🎵 PRUEBA DE BÚSQUEDA DE AUDIO")
    print("="*60)
    
    # Inicializar motor
    print("\n🚀 Cargando sistema...")
    engine = AudioSearchEngine()
    engine.load_index()
    
    print("\n" + "="*60)
    print("Sistema listo. Tienes", len(engine.metadata), "audios en la base de datos")
    print("="*60)
    
    # Mostrar algunos audios disponibles
    print("\n📋 Algunos audios disponibles:")
    for i in range(min(5, len(engine.metadata))):
        print(f"   {i}: {engine.metadata[i]['file_name']}")
    
    # Hacer búsqueda con el primer audio
    print("\n🔍 Probando búsqueda con audio ID 0...")
    
    query_path = engine.metadata[0]['file_path']
    print(f"Query: {Path(query_path).name}")
    
    # Búsqueda indexada
    results, time_taken, candidates = engine.search(
        query_path, 
        k=10, 
        method="indexed"
    )
    
    # Mostrar resultados
    print(f"\n⏱️  Tiempo: {time_taken:.4f} segundos")
    print(f"📊 Candidatos evaluados: {candidates}")
    print(f"\n🎵 Top-10 Audios Similares:")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Audio ID: {result['audio_id']}")
        print(f"   Archivo: {result['file_name']}")
        print(f"   Similitud: {result['similarity']:.4f}")
        print(f"   Ruta: {result['file_path']}")
    
    print("\n" + "="*60)
    print("✅ Búsqueda completada exitosamente")
    print("="*60)
    
    # Comparación con secuencial
    print("\n📊 Comparando con búsqueda secuencial...")
    results_seq, time_seq, candidates_seq = engine.search(
        query_path, 
        k=10, 
        method="sequential"
    )
    
    print(f"\n⚡ Comparación:")
    print(f"   Secuencial: {time_seq:.4f}s ({candidates_seq} audios)")
    print(f"   Indexado:   {time_taken:.4f}s ({candidates} audios)")
    
    if time_seq > 0 and time_taken > 0:
        speedup = time_seq / time_taken
        print(f"   Speedup: {speedup:.2f}x más rápido")


def search_audio(audio_id: int, k: int = 10):
    """
    Busca audios similares a un audio específico
    
    Args:
        audio_id: ID del audio a buscar
        k: Número de resultados
    """
    print(f"\n🔍 Buscando audios similares al audio {audio_id}...")
    
    # Cargar motor
    engine = AudioSearchEngine()
    engine.load_index()
    
    # Verificar que existe
    if audio_id not in engine.metadata:
        print(f"❌ Error: Audio {audio_id} no existe")
        return
    
    # Buscar
    query_path = engine.metadata[audio_id]['file_path']
    results, time_taken, _ = engine.search(query_path, k=k, method="indexed")
    
    # Mostrar
    print(f"\n🎵 Top-{k} para: {engine.metadata[audio_id]['file_name']}")
    print(f"⏱️  Tiempo: {time_taken:.4f}s\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['file_name']}: {result['similarity']:.4f}")


if __name__ == "__main__":
    # Ejecutar prueba básica
    test_search()
    
    # Ejemplos adicionales (comentados)
    # search_audio(5, k=10)  # Buscar similares al audio 5
    # search_audio(42, k=20) # Buscar similares al audio 42