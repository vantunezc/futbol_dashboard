import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Radar
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL DE LA PLATAFORMA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plataforma Pro de Análisis Táctico & Scouting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ Plataforma de Análisis Táctico, Machine Learning & Scouting")
st.caption("Datos oficiales de eventos abiertos de StatsBomb | Pipeline en Python, Scikit-Learn y mplsoccer")

# ---------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DE DATOS (CACHED)
# ---------------------------------------------------------
@st.cache_data
def cargar_dataset_partido():
    # Final Copa del Mundo Femenina 2019: USA vs Países Bajos (ID 22912)
    url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json'
    resp = requests.get(url)
    eventos = resp.json()
    df = pd.json_normalize(eventos)
    return df

df_eventos = cargar_dataset_partido()

# ---------------------------------------------------------
# 3. PESTAÑAS PRINCIPALES DEL DASHBOARD
# ---------------------------------------------------------
tab_tiros, tab_xg, tab_radar, tab_red = st.tabs([
    "🎯 Mapa de Tiros Espacial",
    "📈 Comparativa de Modelos xG",
    "📊 Radar Multidimensional (Scouting)",
    "🕸️ Red de Pases Táctica"
])

# =========================================================
# PESTAÑA 1: MAPA DE TIROS
# =========================================================
with tab_tiros:
    st.subheader("Análisis de Finalización y Calidad de Remates")
    
    # Procesar disparos de jugada abierta
    df_tiros = df_eventos[df_eventos['type.name'] == 'Shot'].copy()
    if 'shot.type.name' in df_tiros.columns:
        df_tiros = df_tiros[df_tiros['shot.type.name'] != 'Penalty']
        
    df_tiros['x'] = df_tiros['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df_tiros['y'] = df_tiros['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    df_tiros = df_tiros.dropna(subset=['x', 'y']).copy()
    
    df_tiros['jugador'] = df_tiros['player.name']
    df_tiros['equipo'] = df_tiros['team.name']
    df_tiros['es_gol'] = (df_tiros['shot.outcome.name'] == 'Goal').astype(int)
    df_tiros['resultado'] = df_tiros['es_gol'].apply(lambda g: 'gol' if g == 1 else 'no gol')
    df_tiros['sb_xG'] = df_tiros['shot.statsbomb_xg'].fillna(0.05)
    
    c1, c2 = st.columns(2)
    with c1:
        equipo_tiros = st.selectbox("Equipo:", ["Todos"] + list(df_tiros['equipo'].unique()), key="sb_eq_tiros")
    with c2:
        df_sub = df_tiros if equipo_tiros == "Todos" else df_tiros[df_tiros['equipo'] == equipo_tiros]
        jugador_tiros = st.selectbox("Rematador:", ["Todos"] + list(df_sub['jugador'].unique()), key="sb_jug_tiros")
        
    df_filtrado_tiros = df_sub if jugador_tiros == "Todos" else df_sub[df_sub['jugador'] == jugador_tiros]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Remates Totales", len(df_filtrado_tiros))
    m2.metric("Goles Anotados", int(df_filtrado_tiros['es_gol'].sum()))
    m3.metric("xG Acumulado (StatsBomb)", f"{df_filtrado_tiros['sb_xG'].sum():.2f}")
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#12151a', line_color='#4f5b66')
    fig_tiros, ax_tiros = pitch.draw(figsize=(10, 6.5))
    
    if not df_filtrado_tiros.empty:
        colores = ['#2ecc71' if r == 'gol' else '#e74c3c' for r in df_filtrado_tiros['resultado']]
        tamanos = df_filtrado_tiros['sb_xG'] * 900 + 60
        pitch.scatter(
            df_filtrado_tiros['x'], df_filtrado_tiros['y'],
            c=colores, s=tamanos, edgecolors='white', alpha=0.85, ax=ax_tiros
        )
    st.pyplot(fig_tiros)

# =========================================================
# PESTAÑA 2: MODELO PROPIO xG vs STATSBOMB
# =========================================================
with tab_xg:
    st.subheader("Evaluación de Machine Learning: Regresión Logística Propia vs StatsBomb")
    
    # Feature Engineering de distancia y ángulo
    goal_x, goal_y = 120.0, 40.3
    post_y1, post_y2 = 36.3, 44.3
    
    df_tiros['distancia'] = np.sqrt((goal_x - df_tiros['x'])**2 + (goal_y - df_tiros['y'])**2)
    
    def calc_angle(x, y):
        a = np.sqrt((goal_x - x)**2 + (post_y1 - y)**2)
        b = np.sqrt((goal_x - x)**2 + (post_y2 - y)**2)
        c = post_y2 - post_y1
        try:
            cos_a = (a**2 + b**2 - c**2) / (2 * a * b)
            return math.acos(np.clip(cos_a, -1.0, 1.0))
        except:
            return 0.0
            
    df_tiros['angulo'] = df_tiros.apply(lambda r: calc_angle(r['x'], r['y']), axis=1)
    
    X = df_tiros[['distancia', 'angulo']]
    y = df_tiros['es_gol']
    
    clf = LogisticRegression()
    clf.fit(X, y)
    df_tiros['mi_xG'] = clf.predict_proba(X)[:, 1]
    df_tiros['delta_xG'] = df_tiros['mi_xG'] - df_tiros['sb_xG']
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Coeficiente Distancia (ML)", f"{clf.coef_[0][0]:.4f}")
    col_m2.metric("Coeficiente Ángulo (ML)", f"{clf.coef_[0][1]:.4f}")
    
    cols_tabla = ['jugador', 'equipo', 'resultado', 'distancia', 'angulo', 'mi_xG', 'sb_xG', 'delta_xG']
    st.dataframe(df_tiros[cols_tabla].style.format({
        'distancia': '{:.1f} m',
        'angulo': '{:.2f} rad',
        'mi_xG': '{:.3f}',
        'sb_xG': '{:.3f}',
        'delta_xG': '{:+.3f}'
    }), width='stretch')

# =========================================================
# PESTAÑA 3: RADAR DE SCOUTING (PERCENTILES REALES PREMIER LEAGUE)
# =========================================================
with tab_radar:
    st.subheader("Perfil Multidimensional y Comparativa por Percentiles (0 a 100)")
    st.caption("Comparación relativa contra todos los jugadores de la Premier League con al menos 270 minutos jugados.")

    from scipy.stats import percentileofscore

    # Cargar dataset local de la liga
    @st.cache_data
    def cargar_datos_scouting():
        try:
            return pd.read_csv("fbref_premier_league.csv")
        except FileNotFoundError:
            return None

    df_liga = cargar_datos_scouting()

    if df_liga is None or df_liga.empty:
        st.warning("⚠️ No se encontró el archivo 'fbref_premier_league.csv'. Ejecuta 'descargar_liga.py' para generarlo.")
    else:
        # Métricas p90 a evaluar en el radar
        metricas_radar = ['Goles_p90', 'xG_p90', 'Asistencias_p90', 'xA_p90']
        metricas_etiquetas = ['Goles p90', 'xG p90', 'Asistencias p90', 'xA p90']
        
        # Selectores de jugadores
        lista_jugadores = sorted(df_liga['Jugador'].unique())
        
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            # Jugador 1 por defecto (si existe)
            idx_default1 = 0
            jugador_1 = st.selectbox("Seleccionar Jugador 1 (Azul):", lista_jugadores, index=idx_default1)
        with c_sel2:
            # Jugador 2 por defecto
            idx_default2 = min(1, len(lista_jugadores) - 1)
            jugador_2 = st.selectbox("Seleccionar Jugador 2 (Naranja):", lista_jugadores, index=idx_default2)

        # Función para calcular el percentil (0-100) de un jugador en cada métrica
        def obtener_percentiles(nombre):
            datos_j = df_liga[df_liga['Jugador'] == nombre].iloc[0]
            percentiles = []
            for col in metricas_radar:
                pct = percentileofscore(df_liga[col], datos_j[col], kind='rank')
                percentiles.append(round(pct, 1))
            return percentiles, datos_j

        pct_j1, fila_j1 = obtener_percentiles(jugador_1)
        pct_j2, fila_j2 = obtener_percentiles(jugador_2)

        # Métricas rápidas de los seleccionados
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**🔵 {jugador_1}** ({fila_j1['Equipo']}) — Minutos: {int(fila_j1['minutes'])}")
        with col_m2:
            st.markdown(f"**🟠 {jugador_2}** ({fila_j2['Equipo']}) — Minutos: {int(fila_j2['minutes'])}")

        # Configuración del radar en mplsoccer (escala universal de percentiles: 0 a 100)
        min_pct = [0.0] * len(metricas_radar)
        max_pct = [100.0] * len(metricas_radar)

        radar = Radar(
            params=metricas_etiquetas,
            min_range=min_pct,
            max_range=max_pct,
            num_rings=4,
            ring_width=1,
            center_circle_radius=1
        )

        fig_r, ax_r = radar.setup_axis(figsize=(7, 7), facecolor='#12151a')
        radar.draw_circles(ax=ax_r, facecolor='#1c2128', edgecolor='#61afef', lw=1.2, alpha=0.35)
        radar.draw_range_labels(ax=ax_r, color='#8b949e', fontsize=9)
        radar.draw_param_labels(ax=ax_r, color='white', fontsize=11, fontweight='bold', offset=1)

        # Dibujar silueta Jugador 1 (Azul cian)
        radar.draw_radar(
            pct_j1, ax=ax_r,
            kwargs_radar={'facecolor': '#00bcd4', 'alpha': 0.45, 'edgecolor': '#00bcd4', 'lw': 2}
        )

        # Dibujar silueta Jugador 2 (Naranja)
        radar.draw_radar(
            pct_j2, ax=ax_r,
            kwargs_radar={'facecolor': '#ff7043', 'alpha': 0.45, 'edgecolor': '#ff7043', 'lw': 2}
        )

        st.pyplot(fig_r)

        # Tabla de percentiles detallada
        df_pct_resumen = pd.DataFrame({
            'Métrica': metricas_etiquetas,
            f'{jugador_1} (Valor Real)': [fila_j1[m] for m in metricas_radar],
            f'{jugador_1} (Percentil)': [f"{p}%" for p in pct_j1],
            f'{jugador_2} (Valor Real)': [fila_j2[m] for m in metricas_radar],
            f'{jugador_2} (Percentil)': [f"{p}%" for p in pct_j2],
        })
        st.dataframe(df_pct_resumen, width='stretch')

# =========================================================
# PESTAÑA 4: RED DE PASES TÁCTICA (PASSING NETWORK)
# =========================================================
with tab_red:
    st.subheader("Estructura Posicional y Conexiones Clave")
    
    equipos_partido = list(df_eventos['team.name'].dropna().unique())
    equipo_sel_red = st.selectbox("Seleccionar Equipo para Red de Pases:", equipos_partido, key="sb_eq_red")
    
    df_equipo_red = df_eventos[df_eventos['team.name'] == equipo_sel_red].copy()
    
    primer_sub = df_equipo_red[df_equipo_red['type.name'] == 'Substitution']['minute'].min()
    min_corte = primer_sub if pd.notna(primer_sub) else 90
    st.caption(f"Evaluando estructura del once titular base (0' al min {int(min_corte)})")
    
    df_pases = df_equipo_red[
        (df_equipo_red['minute'] < min_corte) &
        (df_equipo_red['type.name'] == 'Pass') &
        (df_equipo_red['pass.outcome.name'].isna()) &
        (df_equipo_red['player.name'].notna()) &
        (df_equipo_red['pass.recipient.name'].notna())
    ].copy()
    
    df_pases['x'] = df_pases['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df_pases['y'] = df_pases['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    df_pases = df_pases.dropna(subset=['x', 'y']).copy()
    
    df_pases['pasador'] = df_pases['player.name'].astype(str)
    df_pases['receptor'] = df_pases['pass.recipient.name'].astype(str)
    
    # Posición promedio
    pos_prom = df_pases.groupby('pasador').agg(
        x_media=('x', 'mean'),
        y_media=('y', 'mean'),
        total_pases=('x', 'count')
    ).reset_index()
    
    # Parejas y enlaces
    df_pases['pareja'] = [tuple(sorted([p, r])) for p, r in zip(df_pases['pasador'], df_pases['receptor'])]
    red_df = df_pases.groupby('pareja').size().reset_index(name='conteo')
    red_df = red_df[red_df['conteo'] >= 3].copy()
    
    red_df['jugador1'] = red_df['pareja'].apply(lambda p: str(p[0]))
    red_df['jugador2'] = red_df['pareja'].apply(lambda p: str(p[1]))
    
    red_df = red_df.merge(pos_prom[['pasador', 'x_media', 'y_media']], left_on='jugador1', right_on='pasador', how='inner')
    red_df = red_df.rename(columns={'x_media': 'x1', 'y_media': 'y1'}).drop(columns=['pasador'])
    
    red_df = red_df.merge(pos_prom[['pasador', 'x_media', 'y_media']], left_on='jugador2', right_on='pasador', how='inner')
    red_df = red_df.rename(columns={'x_media': 'x2', 'y_media': 'y2'}).drop(columns=['pasador'])
    
    pitch_red = Pitch(pitch_type='statsbomb', pitch_color='#12151a', line_color='#4f5b66')
    fig_red, ax_red = pitch_red.draw(figsize=(10.5, 7))
    
    # Líneas
    for _, fila in red_df.iterrows():
        ax_red.plot([fila['x1'], fila['x2']], [fila['y1'], fila['y2']], color='#00bcd4', lw=fila['conteo']*0.9, alpha=0.6, zorder=1)
        
    # Nodos
    pitch_red.scatter(pos_prom['x_media'], pos_prom['y_media'], s=pos_prom['total_pases']*20, c='#2ecc71', edgecolors='white', lw=1.5, ax=ax_red, zorder=2)
    
    # Nombres
    for _, fila in pos_prom.iterrows():
        apellido = fila['pasador'].split()[-1]
        pitch_red.annotate(apellido, xy=(fila['x_media'], fila['y_media'] - 2.5), c='white', va='center', ha='center', size=9, weight='bold', ax=ax_red)
        
    st.pyplot(fig_red)

    # Actualiza la declaración de pestañas al inicio:
# tab_tiros, tab_xg, tab_radar, tab_red, tab_xt, tab_clusters = st.tabs([
#     "🎯 Mapa de Tiros Espacial",
#     "📈 Comparativa de Modelos xG",
#     "📊 Radar Multidimensional (Scouting)",
#     "🕸️ Red de Pases Táctica",
#     "⚡ Amenaza Esperada (xT)",
#     "🔍 Arquetipos & Sustitutos (K-Means)"
# ])

# =========================================================
# PESTAÑA 5: AMENAZA ESPERADA (xT)
# =========================================================
with tab_xt:
    st.subheader("Valoración de Acciones Ofensivas: Expected Threat (xT)")
    st.caption("Cuantificación del peligro generado por pases progresivos hacia el último tercio")
    
    # Matriz 12x8
    XT_GRID = np.array([
        [0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.026, 0.038, 0.058, 0.080],
        [0.001, 0.001, 0.002, 0.003, 0.006, 0.009, 0.014, 0.021, 0.031, 0.048, 0.082, 0.125],
        [0.001, 0.002, 0.003, 0.004, 0.007, 0.011, 0.017, 0.026, 0.041, 0.072, 0.155, 0.250],
        [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.019, 0.030, 0.052, 0.105, 0.240, 0.380],
        [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.019, 0.030, 0.052, 0.105, 0.240, 0.380],
        [0.001, 0.002, 0.003, 0.004, 0.007, 0.011, 0.017, 0.026, 0.041, 0.072, 0.155, 0.250],
        [0.001, 0.001, 0.002, 0.003, 0.006, 0.009, 0.014, 0.021, 0.031, 0.048, 0.082, 0.125],
        [0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.026, 0.038, 0.058, 0.080],
    ])
    
    def calc_xt(xs, ys, xe, ye):
        c_s, r_s = int(np.clip(xs / 10.0, 0, 11)), int(np.clip(ys / 10.0, 0, 7))
        c_e, r_e = int(np.clip(xe / 10.0, 0, 11)), int(np.clip(ye / 10.0, 0, 7))
        return max(0.0, XT_GRID[r_e, c_e] - XT_GRID[r_s, c_s])

    df_pases_xt = df_eventos[
        (df_eventos['type.name'] == 'Pass') &
        (df_eventos['pass.outcome.name'].isna()) &
        (df_eventos['player.name'].notna())
    ].copy()
    
    df_pases_xt['xs'] = df_pases_xt['location'].apply(lambda l: l[0] if isinstance(l, list) else None)
    df_pases_xt['ys'] = df_pases_xt['location'].apply(lambda l: l[1] if isinstance(l, list) else None)
    df_pases_xt['xe'] = df_pases_xt['pass.end_location'].apply(lambda l: l[0] if isinstance(l, list) else None)
    df_pases_xt['ye'] = df_pases_xt['pass.end_location'].apply(lambda l: l[1] if isinstance(l, list) else None)
    df_pases_xt = df_pases_xt.dropna(subset=['xs', 'ys', 'xe', 'ye'])
    
    df_pases_xt['xT'] = df_pases_xt.apply(lambda r: calc_xt(r['xs'], r['ys'], r['xe'], r['ye']), axis=1)
    
    rk_xt = df_pases_xt.groupby(['player.name', 'team.name']).agg(
        Pases=('xT', 'count'),
        xT_Total=('xT', 'sum')
    ).reset_index().sort_values(by='xT_Total', ascending=False)
    
    st.dataframe(rk_xt.rename(columns={'player.name': 'Jugador', 'team.name': 'Equipo'}).style.format({'xT_Total': '{:.3f}'}), use_container_width=True)

# =========================================================
# PESTAÑA 6: CLUSTERING Y SUSTITUTOS
# =========================================================
with tab_clusters:
    st.subheader("Buscador de Sustitutos y Arquetipos (Machine Learning)")
    try:
        df_scout = pd.read_csv('fbref_premier_league.csv')
    except Exception:
        df_scout = pd.DataFrame()
        
    if df_scout.empty:
        st.warning("Ejecuta 'python scraping_fbref.py' para contar con el dataset de la Premier League.")
    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        
        cols_metricas = [c for c in df_scout.columns if '_p90' in c]
        if cols_metricas:
            X_scout = StandardScaler().fit_transform(df_scout[cols_metricas].fillna(0))
            
            jugador_buscar = st.selectbox("Seleccionar Jugador de Referencia:", sorted(df_scout['Jugador'].unique()))
            
            idx_obj = df_scout[df_scout['Jugador'] == jugador_buscar].index[0]
            sims = cosine_similarity(X_scout[df_scout.index.get_loc(idx_obj)].reshape(1, -1), X_scout)[0]
            
            df_scout['Similitud Táctica %'] = (sims * 100).round(1)
            sustitutos = df_scout[df_scout['Jugador'] != jugador_buscar].sort_values(by='Similitud Táctica %', ascending=False)
            
            st.write(f"Jugadores estadísticamente más parecidos a **{jugador_buscar}**:")
            cols_mostrar = ['Jugador', 'Equipo', 'Rol', 'Similitud Táctica %'] + cols_metricas
            st.dataframe(sustitutos[cols_mostrar].head(5), use_container_width=True)