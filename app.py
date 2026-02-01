
import streamlit as st

st.set_page_config(
    page_title="TFG | Vehículos Usados CR",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Modelado predictivo y segmentación del mercado de vehículos usados en Costa Rica")
st.markdown("### Inteligencia Artificial con Python")
st.markdown("**Proyecto Final de Graduación – Generación Joan Clarke**")
st.markdown("**Autor:** Ronald Ramírez Espinoza")

st.divider()

st.header("📘 Descripción del Proyecto")

st.markdown("""
El proyecto se desarrollará en el contexto del mercado de vehículos usados en Costa Rica, utilizando como fuente principal de información
los anuncios publicados en el sitio web público crautos.com. Este portal concentra una gran parte de la oferta de vehículos usados del país
y contiene información relevante como marca, modelo, año, precio, kilometraje, tipo de combustible, transmisión, estilo del vehículo, ubicación
 geográfica y características adicionales (equipamiento, extras, etc.).
            
El entorno general de los datos corresponde al mercado automotriz y, en particular, a la oferta de vehículos usados. Se recolectará un historial
de anuncios mediante técnicas de web scraping controlado. Se espera construir una tabla donde cada fila representa un vehículo anunciado en el sitio.
Cada registro contendrá entre 10 y 18 variables relacionadas con las características del vehículo y del anuncio (marca, modelo, año, precio, kilometraje,
provincia, tipo de combustible, transmisión, estilo, etc.).
            
El problema general por analizar es la estimación del precio de mercado de un vehículo usado en Costa Rica en función de sus características, así como
la segmentación del mercado automotriz nacional en grupos de vehículos con perfiles similares. Desde el punto de vista de negocio, esto aporta valor tanto
a compradores (para saber si un precio es razonable) como a vendedores (para fijar precios competitivos) y a posibles intermediarios (por ejemplo, concesionarios
o plataformas de valoración).Desde el punto de vista técnico, el proyecto permitirá aplicar diversos métodos de aprendizaje supervisado y no supervisado
estudiados en el programa.

En la parte supervisada se construirá un modelo de regresión para predecir el precio del vehículo, utilizando algoritmos como regresión lineal y sus variantes
(Ridge, LASSO), árboles de decisión, bosques aleatorios, métodos de potenciación (boosting), máquinas de soporte vectorial (SVM),K vecinos más cercanos (KNN) y
redes neuronales (incluyendo algún modelo de Deep Learning sencillo para regresión). En la parte no supervisada se aplicarán técnicas de clustering (K-medias,
 agrupación jerárquica) y Análisis de Componentes Principales (ACP) para reducir la dimensionalidad y visualizar mejor la estructura del mercado de vehículos.

El desarrollo sigue explícitamente la metodología CRISP-DM.
""")

st.header("🎯 Objetivo General")

st.markdown("""
Desarrollar un sistema de inteligencia artificial que permita predecir el precio 
de vehículos usados en Costa Rica y segmentar el mercado automotriz 
en grupos de vehículos con características similares.
""")

st.header("📌 Objetivos Específicos")

st.markdown("""
- Recolectar, limpiar y estructurar un conjunto de datos de vehículos usados.
- Construir y comparar múltiples modelos de regresión supervisada.
- Aplicar técnicas de clustering y reducción de dimensionalidad.
- Evaluar los modelos mediante métricas apropiadas (MAE, RMSE, R²).
- Desarrollar un demo funcional.
""")

st.header("🧠 Enfoque Metodológico")

st.markdown("""
**Aprendizaje Supervisado**
- Regresión lineal y regularizada
- Árboles de decisión
- Bosques aleatorios
- Métodos de potenciación
- SVM, KNN y redes neuronales

**Aprendizaje No Supervisado**
- K-medias
- Agrupación jerárquica
- PCA

El proyecto sigue la metodología CRISP-DM.
""")

st.header("📂 Estructura de la Aplicación")

st.markdown("""
- Business Case  
- Análisis Exploratorio  
- Modelado Predictivo  
- Segmentación  
- Demo interactivo  
""")

st.divider()
st.info("Utilice el menú lateral para navegar por las diferentes secciones del proyecto.")
