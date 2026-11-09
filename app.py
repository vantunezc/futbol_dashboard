import streamlit as st
import pandas as pd
import requests

@st.cache_data
def cargar_datos_reales():
    # Descarga directa del JSON de eventos de StatsBomb
    url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/22912.json'
    respuesta = requests.get(url)
    eventos = respuesta.json()
    
    # Aplanar estructura JSON
    df_eventos = pd.json_normalize(eventos)
    
    # Filtrar únicamente disparos
    df_tiros = df_eventos[df_eventos['type.name'] == 'Shot'].copy()
    
    # Extraer coordenadas [x, y]
    df_tiros['x'] = df_tiros['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df_tiros['y'] = df_tiros['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    
    # Estandarizar columnas clave
    df_tiros['jugador'] = df_tiros['player.name']
    df_tiros['equipo'] = df_tiros['team.name']
    df_tiros['resultado'] = df_tiros['shot.outcome.name'].apply(lambda r: 'gol' if r == 'Goal' else 'no gol')
    df_tiros['xG'] = df_tiros['shot.statsbomb_xg'].fillna(0.05)
    
    return df_tiros.dropna(subset=['x', 'y'])