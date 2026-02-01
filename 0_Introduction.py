import streamlit as st

st.set_page_config(
    page_title="Proyecto | TFG Vehículos Usados"
)

st.set_page_config(
    page_title="TFG | Vehículos Usados CR",
    page_icon="🚗",
    layout="centered"
)

# -----------------------------
# HERO / PORTADA
# -----------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🚗 Modelado predictivo y segmentación del mercado de vehículos usados en Costa Rica")
    st.caption("Aplicación basada en CRISP-DM | Datos públicos de Crautos.com")

with col2:
    st.markdown("### 📌")
    st.markdown("**Inteligencia Artificial con Python**")
    st.markdown("**TFG – Generación Joan Clarke**")
    st.markdown("**Autor:** Ronald Ramírez Espinoza")

st.divider()

# -----------------------------
# CONTENIDO EN TABS
# -----------------------------
tab_desc, tab_obj, tab_estructura = st.tabs(["📘 Proyecto", "🎯 Objetivos", "📂 Estructura"])

with tab_desc:
    st.header("📘 Descripción del Proyecto")

    st.markdown("""
**Contexto y fuente de datos**  
El proyecto se desarrolla en el mercado de vehículos usados en Costa Rica, utilizando como fuente principal los anuncios publicados en el sitio web público **Crautos.com**.  
Este portal concentra una parte significativa de la oferta nacional e incluye información como **marca, modelo, año, precio, kilometraje, combustible, transmisión, estilo, ubicación** y características adicionales (extras/equipamiento).

**Datos y recolección**  
Se recolecta un historial de anuncios mediante **web scraping controlado**.  
La tabla resultante representa vehículos anunciados, donde cada fila corresponde a un vehículo y cada registro contiene aproximadamente **10 a 18 variables** relevantes (marca, modelo, año, precio, kilometraje, provincia, combustible, transmisión, estilo, etc.).

**Problema a resolver y valor de negocio**  
Se aborda:
- La **estimación del precio de mercado** de un vehículo usado según sus características.
- La **segmentación del mercado** en grupos de vehículos con perfiles similares.

Esto aporta valor a:
- **Compradores:** evaluar si un precio es razonable.
- **Vendedores:** definir precios competitivos.
- **Intermediarios:** concesionarios o plataformas de valoración.

**Enfoque técnico**  
- **Supervisado:** modelo de regresión para predecir precio (regresión lineal y regularizada, árboles, random forest, boosting, SVM, KNN, redes neuronales).
- **No supervisado:** clustering (K-medias, jerárquico) y **ACP/PCA** para reducir dimensionalidad e interpretar la estructura del mercado.

El desarrollo sigue explícitamente la metodología **CRISP-DM**.
""")

with tab_obj:
    st.header("🎯 Objetivo General")
    st.success("""
Desarrollar un sistema de inteligencia artificial, implementado en Python, que permita **predecir el precio** de vehículos usados en Costa Rica y **segmentar el mercado automotriz**
en grupos de vehículos con características similares, utilizando datos recolectados de Crautos.com y aplicando métodos supervisados y no supervisados.
""")

    st.header("📌 Objetivos Específicos")
    st.markdown("""
1. **Recolectar, limpiar y estructurar** un conjunto de datos de vehículos usados anunciados en Crautos.com, construyendo una tabla con variables relevantes (marca, modelo, año, kilometraje, ubicación, características técnicas y precio).
2. **Construir y comparar** modelos de regresión supervisada (regresión lineal y regularizada, árboles de decisión, bosques aleatorios, boosting, SVM, KNN y redes neuronales) para predecir el precio, evaluando desempeño con **MAE, RMSE y R²**.
3. **Aplicar aprendizaje no supervisado** (ACP/PCA, K-medias y agrupación jerárquica) para segmentar el mercado e interpretar perfiles que aporten valor al análisis del mercado costarricense.
""")

with tab_estructura:
    st.header("📂 Estructura de la Aplicación")
    st.markdown("""
Esta aplicación está organizada en secciones (páginas) para documentar y demostrar el proyecto:

- 📌 **Business Case**
- 🕷️ **Web Scraping**
- 📊 **Análisis Exploratorio**
- 📈 **Segmentación de Mercado (Unsupervised Learning)**
- 🤖 **Predicción de valor de mercado (Supervised Learning)**
- ✅ **Conclusiones**
""")

    st.warning("Nota: Streamlit detecta páginas automáticamente usando la carpeta `pages/`.")

st.divider()
st.info("Utilice el menú lateral para navegar por las diferentes secciones del proyecto.")