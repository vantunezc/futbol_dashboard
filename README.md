# ⚽ Plataforma Integral de Análisis Táctico, Scouting & Modelado xG

Dashboard interactivo y pipeline analítico desarrollado en Python para la evaluación espacial de rendimiento, modelado probabilístico de Expected Goals (xG) y caracterización táctica de posesión.

🔗 **Demo en vivo:** [https://TU_USUARIO-futbol-dashboard.streamlit.app](https://TU_USUARIO-futbol-dashboard.streamlit.app)

---

## 🚀 Características del Proyecto
1. **Mapa de Tiros Espacial:** Visualización de remates en campo oficial StatsBomb (`mplsoccer`), discriminando goles, atajadas y volumen de peligro.
2. **Modelo Propio de xG:** Algoritmo entrenado con Regresión Logística (`scikit-learn`) a partir de la distancia euclidiana y el ángulo visible de tiro, comparado tiro a tiro contra el modelo oficial de StatsBomb.
3. **Radar de Scouting:** Comparación multidimensional de perfiles tácticos normalizados por 90 minutos (`p90`).
4. **Red de Pases Táctica (*Passing Networks*):** Estructura del once titular base previo a sustituciones, calculando coordenadas medias y peso de conexiones entre futbolistas.
5. **Módulo de Ingesta:** Extracción de datos tabulares agregados mediante scraping ético (`BeautifulSoup4`).

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Procesamiento de Datos:** Pandas, NumPy
* **Ciencia de Datos / ML:** Scikit-Learn
* **Visualización Táctica:** Matplotlib, mplsoccer
* **Despliegue e Interactividad:** Streamlit Community Cloud
* **Control de Versiones:** Git & GitHub

---

## 💻 Instalación Local
```bash
git clone [https://github.com/vantunezc/futbol_dashboard.git](https://github.com/TU_USUARIO/futbol_dashboard.git)
cd futbol_dashboard
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py