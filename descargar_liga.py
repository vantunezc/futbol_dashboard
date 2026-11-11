import pandas as pd
import requests

print("[*] Conectando con el repositorio de datos abiertos de Premier League...")

# Dataset abierto consolidado con métricas de partidos, minutos, goles, asistencias y xG
url_datos = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"

try:
    print("[*] Descargando registros de la temporada...")
    df = pd.read_csv(url_datos, encoding='utf-8')
    
    # Agrupar por jugador y equipo para tener métricas acumuladas de la temporada
    df_resumen = df.groupby(['name', 'team']).agg({
        'minutes': 'sum',
        'goals_scored': 'sum',
        'assists': 'sum',
        'expected_goals': 'sum',
        'expected_assists': 'sum',
        'clean_sheets': 'sum'
    }).reset_index()

    # Filtrar jugadores con al menos 270 minutos (3 partidos completos)
    df_resumen = df_resumen[df_resumen['minutes'] >= 270].copy()

    # Normalizar métricas por cada 90 minutos (p90)
    df_resumen['90s'] = df_resumen['minutes'] / 90.0
    df_resumen['Goles_p90'] = (df_resumen['goals_scored'] / df_resumen['90s']).round(2)
    df_resumen['Asistencias_p90'] = (df_resumen['assists'] / df_resumen['90s']).round(2)
    df_resumen['xG_p90'] = (df_resumen['expected_goals'] / df_resumen['90s']).round(2)
    df_resumen['xA_p90'] = (df_resumen['expected_assists'] / df_resumen['90s']).round(2)

    # Renombrar columnas para estandarizar con el formato de scouting
    df_resumen = df_resumen.rename(columns={'name': 'Jugador', 'team': 'Equipo'})

    # Guardar en el archivo CSV local
    archivo_salida = "fbref_premier_league.csv"
    df_resumen.to_csv(archivo_salida, index=False, encoding='utf-8')

    print(f"[+] Archivo '{archivo_salida}' generado exitosamente con {len(df_resumen)} jugadores.")
    print(df_resumen[['Jugador', 'Equipo', 'minutes', 'Goles_p90', 'xG_p90', 'xA_p90']].head(6))

except Exception as e:
    print(f"[!] Error al procesar el dataset: {e}")