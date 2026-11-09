import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Radar

# 1. Configuración general de la página
st.set_page_config(page_title="Dashboard Profesional de Fútbol", layout="wide")
st.title("⚽ Plataforma de Análisis Táctico & Scouting")

# 2. Carga y normalización de datos reales de StatsBomb Open Data
@st.cache_data
def cargar_datos_statsbomb():
    # Final Mundial Femenino 2019: USA vs Holanda (ID de evento 22912)
    url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json'
    respuesta = requests.get(url)
    eventos = respuesta.json()
    
    # Aplanar el JSON
    df = pd.json_normalize(eventos)
    
    # Filtrar tiros de juego abierto (excluyendo penales)
    df_tiros = df[df['type.name'] == 'Shot'].copy()
    if 'shot.type.name' in df_tiros.columns:
        df_tiros = df_tiros[df_tiros['shot.type.name'] != 'Penalty']
        
    # Extraer coordenadas X e Y
    df_tiros['x'] = df_tiros['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df_tiros['y'] = df_tiros['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    df_tiros = df_tiros.dropna(subset=['x', 'y'])
    
    # Mapear columnas clave
    df_tiros['jugador'] = df_tiros['player.name']
    df_tiros['equipo'] = df_tiros['team.name']
    df_tiros['resultado'] = df_tiros['shot.outcome.name'].apply(lambda r: 'gol' if r == 'Goal' else 'no gol')
    df_tiros['xG'] = df_tiros['shot.statsbomb_xg'].fillna(0.05)
    
    return df_tiros

df_tiros = cargar_datos_statsbomb()

# 3. Pestañas de navegación
tab_tiros, tab_radar = st.tabs(["🎯 Mapa de Tiros (StatsBomb)", "📊 Comparativa Radar (Scouting)"])

# ==========================================
# PESTAÑA 1: MAPA DE TIROS CON DATOS REALES
# ==========================================
with tab_tiros:
    st.subheader("Análisis Espacial de Tiros Oficiales")
    
    col_filtro1, col_filtro2 = st.columns(2)
    with col_filtro1:
        equipos = ["Todos"] + list(df_tiros['equipo'].unique())
        equipo_sel = st.selectbox("Filtrar por Equipo:", equipos)
        
    with col_filtro2:
        if equipo_sel != "Todos":
            jugadores = ["Todos"] + list(df_tiros[df_tiros['equipo'] == equipo_sel]['jugador'].unique())
        else:
            jugadores = ["Todos"] + list(df_tiros['jugador'].unique())
        jugador_sel = st.selectbox("Filtrar por Rematador:", jugadores)
        
    # Filtrado dinámico
    df_vista = df_tiros.copy()
    if equipo_sel != "Todos":
        df_vista = df_vista[df_vista['equipo'] == equipo_sel]
    if jugador_sel != "Todos":
        df_vista = df_vista[df_vista['jugador'] == jugador_sel]
        
    # Métricas clave en tarjetas
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Remates", len(df_vista))
    m2.metric("Goles Anotados", len(df_vista[df_vista['resultado'] == 'gol']))
    m3.metric("xG Acumulado", f"{df_vista['xG'].sum():.2f}")
    
    # Cancha de tiro con mplsoccer
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#101010', line_color='#7c7c7c')
    fig, ax = pitch.draw(figsize=(10, 6.5))
    
    if not df_vista.empty:
        colores = ['#2ecc71' if r == 'gol' else '#e74c3c' for r in df_vista['resultado']]
        tamanos = df_vista['xG'] * 900 + 60
        pitch.scatter(
            df_vista['x'], df_vista['y'],
            c=colores, s=tamanos, edgecolors='white',
            alpha=0.85, ax=ax
        )
    st.pyplot(fig)

# ==========================================
# PESTAÑA 2: GRÁFICO RADAR COMPARATIVO
# ==========================================
with tab_radar:
    st.subheader("Comparativa Multidimensional de Perfiles")
    
    parametros = [
        'Goles p90', 'xG p90', 'Asistencias p90', 
        'Pases Clave p90', 'Regates p90', 'Duelos Ganados %'
    ]
    min_valores = [0.0, 0.0, 0.0, 0.5, 0.5, 30.0]
    max_valores = [0.8, 0.7, 0.6, 4.0, 5.0, 75.0]
    
    # Datos de muestra configurables
    perfil_a = [0.45, 0.40, 0.35, 2.8, 3.2, 58.0]
    perfil_b = [0.20, 0.25, 0.50, 3.6, 1.8, 68.0]
    
    radar = Radar(
        params=parametros,
        min_range=min_valores,
        max_range=max_valores,
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )
    
    fig_radar, ax_radar = radar.setup_axis(figsize=(7, 7), facecolor='#1b1e23')
    radar.draw_circles(ax=ax_radar, facecolor='#282c34', edgecolor='#61afef', lw=1.2, alpha=0.3)
    radar.draw_range_labels(ax=ax_radar, color='#abb2bf', fontsize=9)
    radar.draw_param_labels(ax=ax_radar, color='white', fontsize=10, fontweight='bold', offset=1)
    
    # Silueta Perfil A (Cian)
    radar.draw_radar(
        perfil_a, ax=ax_radar,
        kwargs_radar={'facecolor': '#00bcd4', 'alpha': 0.4, 'edgecolor': '#00bcd4', 'lw': 2}
    )
    # Silueta Perfil B (Naranja)
    radar.draw_radar(
        perfil_b, ax=ax_radar,
        kwargs_radar={'facecolor': '#ff5722', 'alpha': 0.4, 'edgecolor': '#ff5722', 'lw': 2}
    )
    
    st.pyplot(fig_radar)
    st.markdown("**Leyenda:** 🔵 **Perfil Ofensivo / Extremo** (Azul) vs 🟠 **Perfil Mixto / Creador** (Naranja)")