import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

# ==============================================================================
# 1. CARGA O SIMULACIÓN DE DATOS
# ==============================================================================
# Si tienes tu propio DataFrame, cárgalo aquí (descomentando tu lectura):
# df = pd.read_csv("eventos_partido.csv")


def generar_datos_ejemplo():
    """Genera un DataFrame sintético con formato estándar (StatsBomb 120x80)."""
    np.random.seed(42)
    titulares = [
        "1. Ter Stegen",
        "23. Kounde",
        "4. Araujo",
        "15. Christensen",
        "3. Balde",
        "21. De Jong",
        "8. Pedri",
        "6. Gavi",
        "7. Dembele",
        "9. Lewandowski",
        "22. Raphinha",
    ]

    registros = []
    for _ in range(350):
        p1, p2 = np.random.choice(titulares, size=2, replace=False)
        exito = np.random.choice([True, False], p=[0.82, 0.18])
        minuto = np.random.randint(1, 90)
        x = np.random.uniform(10, 110)
        y = np.random.uniform(10, 70)
        registros.append(
            {
                "minute": minuto,
                "team": "FC Barcelona",
                "type": "Pass",
                "outcome": "Successful" if exito else "Incomplete",
                "player": p1,
                "pass_recipient": p2 if exito else None,
                "x": x,
                "y": y,
            }
        )

    # Simular sustitución al minuto 62
    registros.append(
        {
            "minute": 62,
            "team": "FC Barcelona",
            "type": "Substitution",
            "outcome": None,
            "player": "7. Dembele",
            "pass_recipient": "10. Ansu Fati",
            "x": 0,
            "y": 0,
        }
    )
    return pd.DataFrame(registros)


df = generar_datos_ejemplo()

# ==============================================================================
# 2. FILTRADO METODOLÓGICO (11 TITULAR HASTA PRIMER CAMBIO)
# ==============================================================================
# Identificar minuto de la primera sustitución
primer_cambio = df[df["type"] == "Substitution"]["minute"].min()
minuto_corte = primer_cambio if pd.notnull(primer_cambio) else df["minute"].max()

# Filtrar pases completados del equipo antes del primer cambio
pases = df[
    (df["type"] == "Pass")
    & (df["outcome"] == "Successful")
    & (df["minute"] < minuto_corte)
    & (df["player"].notna())
    & (df["pass_recipient"].notna())
].copy()

# ==============================================================================
# 3. HOMOGENEIZACIÓN DE TIPOS Y CÁLCULO DE POSICIONES MEDIAS
# ==============================================================================
# Garantizar tipo string para evitar: ValueError: merge on float64 and str
pases["player"] = pases["player"].astype(str).str.strip()
pases["pass_recipient"] = pases["pass_recipient"].astype(str).str.strip()

# Calcular coordenadas medias de cada jugador (nodos)
pos_promedio = (
    pases.groupby("player")
    .agg(x_media=("x", "mean"), y_media=("y", "mean"), total_pases=("x", "count"))
    .reset_index()
    .rename(columns={"player": "pasador"})
)

# ==============================================================================
# 4. MATRIZ DE CONEXIONES Y MERGE DE COORDENADAS
# ==============================================================================
# Agrupar pases entre parejas
conexiones = (
    pases.groupby(["player", "pass_recipient"])
    .size()
    .reset_index(name="num_pases")
    .rename(columns={"player": "jugador1", "pass_recipient": "jugador2"})
)

# Filtrar parejas con un umbral mínimo de pases (reduce ruido visual)
UMBRAL_MIN_PASES = 3
conexiones = conexiones[conexiones["num_pases"] >= UMBRAL_MIN_PASES].copy()

# Merge para coordenadas de origen (jugador1)
conexiones = (
    conexiones.merge(
        pos_promedio[["pasador", "x_media", "y_media"]],
        left_on="jugador1",
        right_on="pasador",
        how="inner",
    )
    .rename(columns={"x_media": "x1", "y_media": "y1"})
    .drop(columns=["pasador"])
)

# Merge para coordenadas de destino (jugador2)
conexiones = (
    conexiones.merge(
        pos_promedio[["pasador", "x_media", "y_media"]],
        left_on="jugador2",
        right_on="pasador",
        how="inner",
    )
    .rename(columns={"x_media": "x2", "y_media": "y2"})
    .drop(columns=["pasador"])
)

# ==============================================================================
# 5. VISUALIZACIÓN TÁCTICA CON MPLSOCCER
# ==============================================================================
pitch = Pitch(
    pitch_type="statsbomb",
    pitch_color="#0e1117",
    line_color="#4f5b66",
    linewidth=1.2,
    goal_type="box",
)

fig, ax = pitch.draw(figsize=(13, 8.5))
fig.patch.set_facecolor("#0e1117")

# Normalizar anchos de línea según frecuencia de pase
max_pases = conexiones["num_pases"].max() if len(conexiones) > 0 else 1
conexiones["width"] = (conexiones["num_pases"] / max_pases) * 6

# Dibujar líneas de conexión
pitch.lines(
    conexiones["x1"],
    conexiones["y1"],
    conexiones["x2"],
    conexiones["y2"],
    lw=conexiones["width"],
    color="#00d2be",
    alpha=0.6,
    zorder=2,
    ax=ax,
)

# Escalar tamaño de nodos según volumen total de pases
max_nodo = pos_promedio["total_pases"].max()
pos_promedio["node_size"] = (pos_promedio["total_pases"] / max_nodo) * 900 + 300

# Dibujar nodos de los jugadores
pitch.scatter(
    pos_promedio["x_media"],
    pos_promedio["y_media"],
    s=pos_promedio["node_size"],
    color="#ff0055",
    edgecolors="#ffffff",
    linewidth=2,
    zorder=3,
    alpha=0.95,
    ax=ax,
)

# Añadir nombres o dorsales sobre los nodos
for _, row in pos_promedio.iterrows():
    # Muestra el nombre corto o dorsal para legibilidad
    nombre_mostrar = row["pasador"].split(". ")[-1] if ". " in row["pasador"] else row["pasador"]
    pitch.annotate(
        nombre_mostrar,
        xy=(row["x_media"], row["y_media"] - 3.2),
        c="#ffffff",
        va="top",
        ha="center",
        size=9,
        weight="bold",
        zorder=4,
        ax=ax,
    )

# Títulos y metadatos del gráfico
ax.set_title(
    f"Red de Pases Táctica (Hasta min {minuto_corte})\n"
    f"Conexiones activas: >= {UMBRAL_MIN_PASES} pases completados",
    fontsize=14,
    color="#ffffff",
    pad=15,
    weight="bold",
)

plt.tight_layout()
plt.show()