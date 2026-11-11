import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

def extraer_tabla_fbref(url):
    # Cabeceras completas para simular un navegador de escritorio legítimo
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    print(f"[*] Consultando URL: {url}")
    # Pausa de seguridad
    espera = random.uniform(3.0, 5.0)
    print(f"[*] Esperando {espera:.1f} segundos...")
    time.sleep(espera)
    
    session = requests.Session()
    resp = session.get(url, headers=headers)
    
    if resp.status_code != 200:
        print(f"[!] Error al acceder: Código {resp.status_code}")
        return None
        
    soup = BeautifulSoup(resp.content, 'html.parser')
    tabla = soup.find('table')
    
    if not tabla:
        print("[!] No se encontró ninguna tabla en el documento.")
        return None
        
    df = pd.read_html(str(tabla))[0]
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns.values]
        
    return df

if __name__ == "__main__":
    url_liga = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'
    datos_liga = extraer_tabla_fbref(url_liga)
    
    if datos_liga is not None:
        print(f"[+] Datos extraídos exitosamente: {datos_liga.shape[0]} filas.")
        print(datos_liga.head(5))
        datos_liga.to_csv("fbref_premier_league.csv", index=False)
        print("[+] Guardado en fbref_premier_league.csv")