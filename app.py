
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
Este proyecto desarrolla un sistema de inteligencia artificial implementado en Python, 
orientado al análisis del mercado de vehículos usados en Costa Rica, utilizando datos 
recolectados desde el portal público Crautos.com.

El enfoque combina técnicas de aprendizaje supervisado y no supervisado con el objetivo de:

- Estimar el precio de mercado de un vehículo usado a partir de sus características.
- Analizar la estructura del mercado automotriz nacional.
- Identificar segmentos de vehículos con perfiles similares.

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
