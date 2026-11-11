import pandas as pd
import numpy as np
import requests
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# 1. Cargar múltiples partidos de StatsBomb Open Data para tener más volumen
urls_partidos = [
    'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json',  # Final Mundial Femenino 2019
    'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/18236.json',  # Semifinal Mundial 2018
    'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/18237.json'   # Semifinal Mundial 2018
]

todos_los_tiros = []

for url in urls_partidos:
    res = requests.get(url)
    eventos = res.json()
    df = pd.json_normalize(eventos)
    
    # Filtrar tiros de jugada abierta
    df_tiros = df[df['type.name'] == 'Shot'].copy()
    if 'shot.type.name' in df_tiros.columns:
        df_tiros = df_tiros[df_tiros['shot.type.name'] != 'Penalty']
        
    todos_los_tiros.append(df_tiros)

df_total = pd.concat(todos_los_tiros, ignore_index=True)

# 2. Extraer coordenadas X e Y
df_total['x'] = df_total['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
df_total['y'] = df_total['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
df_total = df_total.dropna(subset=['x', 'y']).copy()

# 3. Feature Engineering: Distancia y Ángulo al arco rival (120, 40.3)
goal_x = 120.0
goal_y = 40.3
post_y1 = 36.3
post_y2 = 44.3

# Distancia euclidiana
df_total['distancia'] = np.sqrt((goal_x - df_total['x'])**2 + (goal_y - df_total['y'])**2)

# Ángulo visible usando ley de cosenos
def calcular_angulo(x, y):
    a = np.sqrt((goal_x - x)**2 + (post_y1 - y)**2)
    b = np.sqrt((goal_x - x)**2 + (post_y2 - y)**2)
    c = post_y2 - post_y1
    try:
        cos_ang = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return math.acos(cos_ang)
    except:
        return 0.0

df_total['angulo'] = df_total.apply(lambda r: calcular_angulo(r['x'], r['y']), axis=1)

# Variable objetivo (1 = Gol, 0 = No Gol)
df_total['es_gol'] = (df_total['shot.outcome.name'] == 'Goal').astype(int)

# 4. Variables predictoras (X) y objetivo (y)
X = df_total[['distancia', 'angulo']]
y = df_total['es_gol']

# 5. Ajustar modelo de Regresión Logística
modelo_xg = LogisticRegression()
modelo_xg.fit(X, y)

# Predecir probabilidades de gol (la columna 1 corresponde a P(Gol))
df_total['mi_xg'] = modelo_xg.predict_proba(X)[:, 1]

# 6. Evaluación del modelo
print("=== RESULTADOS DEL MODELO xG ===")
print(f"Total de tiros procesados: {len(df_total)}")
print(f"Coeficiente Distancia: {modelo_xg.coef_[0][0]:.4f} (negativo indica que a mayor distancia, menor probabilidad)")
print(f"Coeficiente Ángulo:    {modelo_xg.coef_[0][1]:.4f} (positivo indica que a mayor ángulo, mayor probabilidad)")
print(f"Brier Score (error cuadrático medio de calibración): {brier_score_loss(y, df_total['mi_xg']):.4f}")

# Muestra comparativa
columnas_ver = ['player.name', 'distancia', 'angulo', 'es_gol', 'mi_xg', 'shot.statsbomb_xg']
print("\nComparación de primeros tiros:")
print(df_total[columnas_ver].head(6))