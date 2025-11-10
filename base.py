# ### **FASE 1: Extracción de Descriptores Acústicos**
# # Ejemplo conceptual
# import librosa
# mfcc = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=13)
# ```

# ---

# ### **FASE 2: Construcción del Diccionario Acústico (Codebook)**

# **2.1 Recolección de Descriptores**
# - Reunir todos los descriptores MFCC de tu colección de audios
# - Crear un gran conjunto de vectores

# **2.2 Clustering con K-Means**
# - Aplicar K-Means sobre todos los descriptores
# - Cada cluster = una "acoustic word"
# - Los centroides = codewords del diccionario
# - Elegir valor de K (ej: 500, 1000, 2000 palabras acústicas)

# ---

# ### **FASE 3: Representación como Histogramas**

# **3.1 Bag of Acoustic Words**
# - Para cada audio, asignar sus descriptores al codeword más cercano
# - Crear histograma: frecuencia de cada acoustic word
# - Resultado: cada audio = vector de dimensión K

# **3.2 Ponderación TF-IDF**
# - Calcular TF (frecuencia del término en el audio)
# - Calcular IDF (importancia del término en la colección)
# - Aplicar ponderación TF-IDF a cada palabra acústica

# ---

# ### **FASE 4: Construcción del Índice Invertido Flat**

# **4.1 Estructura del Índice**
# ```
# acoustic_word_1 -> [(audio_id_1, tf-idf_score), (audio_id_5, tf-idf_score), ...]
# acoustic_word_2 -> [(audio_id_2, tf-idf_score), (audio_id_7, tf-idf_score), ...]
# ...


