import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Radar
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Dashboard Profesional de Fútbol", layout="wide")
st.title("⚽ Plataforma de Análisis Táctico, Scouting & Modelado xG")

# 1. Carga y cálculo matemático de variables
@st.cache_data
def cargar_y_modelar_datos():
    url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json'
    respuesta = requests.get(url)
    df = pd.json_normalize(respuesta.json())
    
    # Filtrar jugada abierta
    df_tiros = df[df['type.name'] == 'Shot'].copy()
    if 'shot.type.name' in df_tiros.columns:
        df_tiros = df_tiros[df_tiros['shot.type.name'] != 'Penalty']
        
    df_tiros['x'] = df_tiros['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df_tiros['y'] = df_tiros['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    df_tiros = df_tiros.dropna(subset=['x', 'y']).copy()
    
    # Coordenadas de arco
    goal_x, goal_y = 120.0, 40.3
    post_y1, post_y2 = 36.3, 44.3
    
    # Distancia euclidiana y ángulo
    df_tiros['distancia'] = np.sqrt((goal_x - df_tiros['x'])**2 + (goal_y - df_tiros['y'])**2)
    
    def get_angle(x, y):
        a = np.sqrt((goal_x - x)**2 + (post_y1 - y)**2)
        b = np.sqrt((goal_x - x)**2 + (post_y2 - y)**2)
        c = post_y2 - post_y1
        try:
            cos_a = (a**2 + b**2 - c**2) / (2 * a * b)
            return math.acos(np.clip(cos_a, -1.0, 1.0))
        except:
            return 0.0
            
    df_tiros['angulo'] = df_tiros.apply(lambda r: get_angle(r['x'], r['y']), axis=1)
    df_tiros['es_gol'] = (df_tiros['shot.outcome.name'] == 'Goal').astype(int)
    
    # Entrenar Regresión Logística
    X = df_tiros[['distancia', 'angulo']]
    y = df_tiros['es_gol']
    
    clf = LogisticRegression()
    clf.fit(X, y)
    df_tiros['mi_xG'] = clf.predict_proba(X)[:, 1]
    
    # Estandarización de nombres
    df_tiros['jugador'] = df_tiros['player.name']
    df_tiros['equipo'] = df_tiros['team.name']
    df_tiros['resultado'] = df_tiros['es_gol'].apply(lambda g: 'gol' if g == 1 else 'no gol')
    df_tiros['sb_xG'] = df_tiros['shot.statsbomb_xg'].fillna(0.05)
    
    return df_tiros

df_tiros = cargar_y_modelar_datos()

# 2. Navegación en pestañas
tab_tiros, tab_comparacion_xg, tab_radar = st.tabs([
    "🎯 Mapa de Tiros", 
    "📈 Comparativa de Modelos (Mi xG vs StatsBomb)", 
    "📊 Comparativa Radar (Scouting)"
])

# ==========================================
# PESTAÑA 1: MAPA DE TIROS
# ==========================================
with tab_tiros:
    st.subheader("Análisis Espacial de Remates")
    col1, col2 = st.columns(2)
    with col1:
        equipo_sel = st.selectbox("Seleccionar Equipo:", ["Todos"] + list(df_tiros['equipo'].unique()))
    with col2:
        df_sub = df_tiros if equipo_sel == "Todos" else df_tiros[df_tiros['equipo'] == equipo_sel]
        jugador_sel = st.selectbox("Seleccionar Jugador:", ["Todos"] + list(df_sub['jugador'].unique()))
        
    df_vista = df_sub if jugador_sel == "Todos" else df_sub[df_sub['jugador'] == jugador_sel]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tiros", len(df_vista))
    m2.metric("Goles", len(df_vista[df_vista['resultado'] == 'gol']))
    m3.metric("Mi xG Acumulado", f"{df_vista['mi_xG'].sum():.2f}")
    m4.metric("StatsBomb xG Acumulado", f"{df_vista['sb_xG'].sum():.2f}")
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#101010', line_color='#7c7c7c')
    fig, ax = pitch.draw(figsize=(10, 6))
    if not df_vista.empty:
        colores = ['#2ecc71' if r == 'gol' else '#e74c3c' for r in df_vista['resultado']]
        tamanos = df_vista['mi_xG'] * 900 + 60
        pitch.scatter(df_vista['x'], df_vista['y'], c=colores, s=tamanos, edgecolors='white', alpha=0.85, ax=ax)
    st.pyplot(fig)

# ==========================================
# PESTAÑA 2: COMPARACIÓN DE MODELOS xG
# ==========================================
with tab_comparacion_xg:
    st.subheader("Discrepancia entre Tu Modelo Logístico y StatsBomb xG")
    st.markdown("Comparativa tiro por tiro evaluando distancia, ángulo y la probabilidad predicha por ambos modelos.")
    
    df_tiros['diferencia_xG'] = df_tiros['mi_xG'] - df_tiros['sb_xG']
    columnas_tabla = ['jugador', 'equipo', 'resultado', 'distancia', 'angulo', 'mi_xG', 'sb_xG', 'diferencia_xG']
    
    st.dataframe(df_tiros[columnas_tabla].style.format({
        'distancia': '{:.1f} m',
        'angulo': '{:.2f} rad',
        'mi_xG': '{:.3f}',
        'sb_xG': '{:.3f}',
        'diferencia_xG': '{:+.3f}'
    }))

# ==========================================
# PESTAÑA 3: RADAR DE SCOUTING
# ==========================================
with tab_radar:
    st.subheader("Perfil Multidimensional de Jugadores")
    parametros = ['Goles p90', 'xG p90', 'Asistencias p90', 'Pases Clave p90', 'Regates p90', 'Duelos Ganados %']
    min_valores = [0.0, 0.0, 0.0, 0.5, 0.5, 30.0]
    max_valores = [0.8, 0.7, 0.6, 4.0, 5.0, 75.0]
    
    perfil_a = [0.45, 0.40, 0.35, 2.8, 3.2, 58.0]
    perfil_b = [0.20, 0.25, 0.50, 3.6, 1.8, 68.0]
    
    radar = Radar(params=parametros, min_range=min_valores, max_range=max_valores, num_rings=4)
    fig_r, ax_r = radar.setup_axis(figsize=(7, 7), facecolor='#1b1e23')
    radar.draw_circles(ax=ax_r, facecolor='#282c34', edgecolor='#61afef', lw=1.2, alpha=0.3)
    radar.draw_range_labels(ax=ax_r, color='#abb2bf', fontsize=9)
    radar.draw_param_labels(ax=ax_r, color='white', fontsize=10, fontweight='bold', offset=1)
    
    radar.draw_radar(perfil_a, ax=ax_r, kwargs_radar={'facecolor': '#00bcd4', 'alpha': 0.4, 'edgecolor': '#00bcd4', 'lw': 2})
    radar.draw_radar(perfil_b, ax=ax_r, kwargs_radar={'facecolor': '#ff5722', 'alpha': 0.4, 'edgecolor': '#ff5722', 'lw': 2})
    st.pyplot(fig_r)