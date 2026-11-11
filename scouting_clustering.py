import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar y filtrar dataset de scouting
df = pd.read_csv('fbref_premier_league.csv')
print(f"[*] Registros cargados: {len(df)}")

# Seleccionar columnas numéricas tácticas
columnas_scouting = [c for c in df.columns if '_p90' in c or c in ['Min', 'minutes']]
df_analisis = df.dropna(subset=['Jugador']).copy()

# Rellenar valores nulos con 0 en métricas
df_analisis[columnas_scouting] = df_analisis[columnas_scouting].fillna(0)

# 2. Estandarización de variables
scaler = StandardScaler()
X_escalado = scaler.fit_transform(df_analisis[columnas_scouting])

# 3. Agrupamiento con K-Means (5 arquetipos tácticos)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_analisis['Cluster'] = kmeans.fit_predict(X_escalado)

# 4. Reducción de dimensionalidad con PCA a 2 componentes para visualización
pca = PCA(n_components=2)
componentes = pca.fit_transform(X_escalado)
df_analisis['PCA1'] = componentes[:, 0]
df_analisis['PCA2'] = componentes[:, 1]

# 5. Función de recomendación: Encontrar jugadores más similares (Similitud Coseno)
def buscar_sustitutos(nombre_jugador, top_n=5):
    if nombre_jugador not in df_analisis['Jugador'].values:
        print(f"[!] Jugador {nombre_jugador} no encontrado.")
        return None
    
    idx = df_analisis[df_analisis['Jugador'] == nombre_jugador].index[0]
    vector_objetivo = X_escalado[df_analisis.index.get_loc(idx)].reshape(1, -1)
    
    similitudes = cosine_similarity(vector_objetivo, X_escalado)[0]
    df_analisis['Similitud'] = (similitudes * 100).round(1)
    
    resultado = df_analisis[df_analisis['Jugador'] != nombre_jugador].sort_values(
        by='Similitud', ascending=False
    )[['Jugador', 'Equipo', 'Rol', 'Similitud', 'Cluster']].head(top_n)
    
    return resultado

# 6. Proyección gráfica de los clusters
plt.figure(figsize=(10, 7), facecolor='#12151a')
ax = plt.axes()
ax.set_facecolor('#1c2128')

scatter = ax.scatter(
    df_analisis['PCA1'], 
    df_analisis['PCA2'], 
    c=df_analisis['Cluster'], 
    cmap='viridis', 
    alpha=0.75, 
    s=65, 
    edgecolors='none'
)

# Destacar algunos nombres representativos
for _, fila in df_analisis.sample(min(8, len(df_analisis)), random_state=42).iterrows():
    ax.text(fila['PCA1'] + 0.1, fila['PCA2'], fila['Jugador'], color='white', fontsize=8)

plt.title("Mapeo de Arquetipos Tácticos (K-Means + PCA)", color='white', fontsize=13)
plt.xlabel("Componente Principal 1", color='white')
plt.ylabel("Componente Principal 2", color='white')
ax.tick_params(colors='white')
plt.grid(True, color='#282c34', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Probar búsqueda con el primer jugador de la lista
primer_jugador = df_analisis['Jugador'].iloc[0]
print(f"\n=== SUSTITUTOS RECOMENDADOS PARA: {primer_jugador} ===")
print(buscar_sustitutos(primer_jugador))