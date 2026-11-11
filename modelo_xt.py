import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# 1. Matriz de transición xT base simplificada (12 columnas x 8 filas)
# Representa la probabilidad empírica de marcar desde cada cuadrante del campo
XT_GRID_12x8 = np.array([
    [0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.026, 0.038, 0.058, 0.080],
    [0.001, 0.001, 0.002, 0.003, 0.006, 0.009, 0.014, 0.021, 0.031, 0.048, 0.082, 0.125],
    [0.001, 0.002, 0.003, 0.004, 0.007, 0.011, 0.017, 0.026, 0.041, 0.072, 0.155, 0.250],
    [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.019, 0.030, 0.052, 0.105, 0.240, 0.380],
    [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.019, 0.030, 0.052, 0.105, 0.240, 0.380],
    [0.001, 0.002, 0.003, 0.004, 0.007, 0.011, 0.017, 0.026, 0.041, 0.072, 0.155, 0.250],
    [0.001, 0.001, 0.002, 0.003, 0.006, 0.009, 0.014, 0.021, 0.031, 0.048, 0.082, 0.125],
    [0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.026, 0.038, 0.058, 0.080],
])

def asignar_celda(x, y, num_cols=12, num_filas=8):
    """Mapea coordenadas continuas de StatsBomb (120x80) a índices de cuadrícula."""
    col = int(np.clip(x / (120.0 / num_cols), 0, num_cols - 1))
    fila = int(np.clip(y / (80.0 / num_filas), 0, num_filas - 1))
    return fila, col

def calcular_valor_xt(x_inicio, y_inicio, x_fin, y_fin):
    """Calcula el delta de amenaza esperada generado por una acción de progresión."""
    f_in, c_in = asignar_celda(x_inicio, y_inicio)
    f_end, c_end = asignar_celda(x_fin, y_fin)
    
    xt_inicio = XT_GRID_12x8[f_in, c_in]
    xt_fin = XT_GRID_12x8[f_end, c_end]
    
    # Solo se premian acciones progresivas hacia adelante
    return max(0.0, xt_fin - xt_inicio)

# 2. Descargar y procesar pases del partido
url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json'
resp = requests.get(url)
df_eventos = pd.json_normalize(resp.json())

# Filtrar pases completados
df_pases = df_eventos[
    (df_eventos['type.name'] == 'Pass') &
    (df_eventos['pass.outcome.name'].isna()) &
    (df_eventos['player.name'].notna())
].copy()

df_pases['x_start'] = df_pases['location'].apply(lambda l: l[0] if isinstance(l, list) else None)
df_pases['y_start'] = df_pases['location'].apply(lambda l: l[1] if isinstance(l, list) else None)
df_pases['x_end'] = df_pases['pass.end_location'].apply(lambda l: l[0] if isinstance(l, list) else None)
df_pases['y_end'] = df_pases['pass.end_location'].apply(lambda l: l[1] if isinstance(l, list) else None)
df_pases = df_pases.dropna(subset=['x_start', 'y_start', 'x_end', 'y_end'])

# Calcular xT generado por cada pase
df_pases['xT'] = df_pases.apply(
    lambda r: calcular_valor_xt(r['x_start'], r['y_start'], r['x_end'], r['y_end']),
    axis=1
)

# 3. Ranking de generadores de amenaza esperada en el partido
ranking_xt = df_pases.groupby(['player.name', 'team.name']).agg(
    pases_completados=('xT', 'count'),
    xt_acumulado=('xT', 'sum'),
    xt_por_pase=('xT', 'mean')
).reset_index().sort_values(by='xt_acumulado', ascending=False)

print("=== TOP 10 JUGADORES CON MAYOR AMENAZA GENERADA (xT) ===")
print(ranking_xt.head(10).to_string(index=False))

# 4. Visualizar mapa de calor de xT sobre el campo
pitch = Pitch(pitch_type='statsbomb', pitch_color='#12151a', line_color='#61afef')
fig, ax = pitch.draw(figsize=(10, 6.5))

# Renderizar cuadrícula con valores de xT
pcm = ax.imshow(
    XT_GRID_12x8, 
    extent=[0, 120, 0, 80], 
    origin='upper', 
    cmap='YlOrRd', 
    alpha=0.45, 
    interpolation='bicubic'
)
cbar = plt.colorbar(pcm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Probabilidad base de amenaza (xT)', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

plt.title("Matriz de Amenaza Esperada (xT) 12x8", color='white', fontsize=13)
plt.show()