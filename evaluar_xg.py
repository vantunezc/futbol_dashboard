import pandas as pd
import numpy as np
import requests
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# 1. Cargar datos de entrenamiento desde StatsBomb Open Data
urls = [
    'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json',
    'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/18236.json',
    'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/18237.json'
]

tiros = []
for url in urls:
    res = requests.get(url)
    df = pd.json_normalize(res.json())
    df_t = df[df['type.name'] == 'Shot'].copy()
    if 'shot.type.name' in df_t.columns:
        df_t = df_t[df_t['shot.type.name'] != 'Penalty']
    tiros.append(df_t)

df_total = pd.concat(tiros, ignore_index=True)
df_total['x'] = df_total['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
df_total['y'] = df_total['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
df_total = df_total.dropna(subset=['x', 'y']).copy()

# 2. Features: Distancia y Ángulo
goal_x, goal_y = 120.0, 40.3
post_y1, post_y2 = 36.3, 44.3

df_total['distancia'] = np.sqrt((goal_x - df_total['x'])**2 + (goal_y - df_total['y'])**2)

def calcular_angulo(x, y):
    a = np.sqrt((goal_x - x)**2 + (post_y1 - y)**2)
    b = np.sqrt((goal_x - x)**2 + (post_y2 - y)**2)
    c = post_y2 - post_y1
    try:
        cos_ang = (a**2 + b**2 - c**2) / (2 * a * b)
        return math.acos(np.clip(cos_ang, -1.0, 1.0))
    except:
        return 0.0

df_total['angulo'] = df_total.apply(lambda r: calcular_angulo(r['x'], r['y']), axis=1)
df_total['es_gol'] = (df_total['shot.outcome.name'] == 'Goal').astype(int)

# 3. Entrenar Regresión Logística
X = df_total[['distancia', 'angulo']]
y = df_total['es_gol']

modelo = LogisticRegression()
modelo.fit(X, y)
df_total['prob_xg'] = modelo.predict_proba(X)[:, 1]

# 4. Métricas de desempeño
brier = brier_score_loss(y, df_total['prob_xg'])
auc = roc_auc_score(y, df_total['prob_xg'])

print(f"Brier Score: {brier:.4f} (Ideal: cercano a 0)")
print(f"ROC-AUC:     {auc:.4f} (0.5 = azar, 1.0 = perfecto)")

# 5. Graficar Curva de Calibración
fraccion_positivos, valor_medio_predicho = calibration_curve(y, df_total['prob_xg'], n_bins=5)

plt.figure(figsize=(7, 5))
plt.plot(valor_medio_predicho, fraccion_positivos, marker='o', label='Modelo xG (Regresión Logística)', color='#3498db', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', label='Calibración Perfecta', color='#7f8c8d')
plt.xlabel('Probabilidad Predicha (xG)')
plt.ylabel('Fracción Real de Goles')
plt.title('Curva de Calibración del Modelo xG')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()