import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

def extraer_tabla_fbref(url):
    """
    Descarga y limpia tablas agregadas de ligas desde FBref simulando un navegador
    para respetar los límites de la tasa de peticiones.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
    }
    
    print(f"[*] Consultando URL: {url}")
    # Pausa ética para evitar rate limiting (HTTP 429)
    espera = random.uniform(2.5, 4.0)
    time.sleep(espera)
    
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"[!] Error al acceder: Código {resp.status_code}")
        return None
        
    # Parsear HTML
    soup = BeautifulSoup(resp.content, 'html.parser')
    tabla = soup.find('table')
    
    if not tabla:
        print("[!] No se encontró ninguna tabla en el documento.")
        return None
        
    df = pd.read_html(str(tabla))[0]
    
    # Aplanar encabezados multinivel si existen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
    return df

if __name__ == "__main__":
    # Ejemplo de uso: Tabla de estadísticas estándar de una liga
    url_liga = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'
    datos_liga = extraer_tabla_fbref(url_liga)
    
    if datos_liga is not None:
        print(f"[+] Datos extraídos exitosamente: {datos_liga.shape[0]} filas.")
        print(datos_liga.head(5))
        datos_liga.to_csv("fbref_premier_league.csv", index=False)
        print("[+] Guardado en fbref_premier_league.csv")