from pathlib import Path

# Tu ruta
FMA_DIR = Path(r"D:\proyecto2bd2\data\fma_small")

print(f"📁 Buscando en: {FMA_DIR}")
print(f"¿Existe?: {FMA_DIR.exists()}")

# Buscar MP3
mp3_files = list(FMA_DIR.rglob("*.mp3"))
print(f"🎵 MP3 encontrados: {len(mp3_files)}")

if len(mp3_files) > 0:
    print(f"\nPrimeros 5:")
    for f in mp3_files[:5]:
        print(f"  ✅ {f.name}")
else:
    print("\n⚠️  NO se encontraron archivos")
