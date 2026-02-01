
import streamlit as st

st.set_page_config(
    page_title="Business Case | TFG Vehículos Usados CR",
    page_icon="📌",
    layout="centered"
)

st.title("📌 1. Comprensión del Negocio (CRISP-DM)")
st.caption(
    "Definición del contexto, objetivos, criterios de éxito, recursos, "
    "supuestos y riesgos del proyecto."
)

st.header("1.1 Determinar los objetivos del negocio")

st.subheader("1.1.1 Background")
st.markdown("""
El mercado de vehículos usados en Costa Rica se caracteriza por una alta heterogeneidad 
en precios, marcas, modelos, antigüedad y kilometraje. Plataformas digitales como 
Crautos.com concentran una parte significativa de la oferta nacional y constituyen 
una fuente relevante de información pública sobre este mercado.

La determinación del precio suele realizarse de manera empírica, mediante comparaciones 
manuales o referencias subjetivas. Esto dificulta evaluar si un precio publicado es 
consistente con el comportamiento general del mercado.

El uso de técnicas de inteligencia artificial y aprendizaje automático permite transformar 
grandes volúmenes de datos en conocimiento estructurado que apoye la toma de decisiones 
relacionadas con la compra, venta y análisis del mercado automotriz.
""")

st.subheader("1.1.2 Objetivos del negocio")
st.markdown("""
- Estimar de manera objetiva el precio de mercado de un vehículo usado en Costa Rica.
- Analizar y segmentar el mercado automotriz costarricense.
- Identificar los principales factores que influyen en la formación de precios.
""")

st.subheader("1.1.3 Criterios de éxito del negocio")
st.markdown("""
El proyecto se considerará exitoso si:

- Identifica factores relevantes en la determinación del precio.
- Genera segmentos interpretables y coherentes con perfiles reales de vehículos.
- Supera referencias triviales de predicción.
- Aporta interpretaciones útiles para la toma de decisiones.

El sistema desarrollado es una herramienta de apoyo y no un mecanismo determinístico 
de fijación de precios.
""")

st.header("1.2 Valoración de la situación")

st.subheader("1.2.1 Inventario de recursos")
st.markdown("""
- Dataset completo extraído desde Crautos.com.
- Python como lenguaje principal.
- Librerías: pandas, numpy, scikit-learn.
- Infraestructura computacional personal.
- Aplicación web en Streamlit para despliegue del demo.
""")

st.subheader("1.2.2 Requisitos, supuestos y restricciones")

st.markdown("**Requisitos**")
st.markdown("""
- Uso exclusivo de datos públicos disponibles en Crautos.com.
- Implementación de modelos directamente en Python.
- Aplicación explícita de la metodología CRISP-DM.
""")

st.markdown("**Supuestos**")
st.markdown("""
- El precio publicado es una aproximación razonable al valor de mercado.
- Las variables disponibles contienen información suficiente para modelar el precio.
""")

st.markdown("**Restricciones**")
st.markdown("""
- No se dispone del precio final de venta.
- El análisis se limita al período cubierto por la extracción.
- La calidad depende de la exactitud de los anuncios.
""")

st.subheader("1.2.3 Riesgos y contingencias")
st.markdown("""
- Presencia de valores atípicos (outliers).
- Registros incompletos o inconsistentes.
- Alta cardinalidad en variables categóricas.

Se aplican técnicas de limpieza, transformación y validación cruzada 
para mitigar estos riesgos.
""")

st.divider()
st.caption("TFG: Analítica del mercado de vehículos usados en Costa Rica | Metodología CRISP-DM")
