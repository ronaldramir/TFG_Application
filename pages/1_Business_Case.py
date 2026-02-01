import streamlit as st

st.set_page_config(
    page_title="Business Case | CRISP-DM",
    page_icon="📌",
    layout="centered"
)

# ------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------

st.title("📌 1. Comprensión del Negocio (CRISP-DM)")
st.caption(
    "Definición del contexto, objetivos y criterios de éxito del proyecto."
)

st.divider()

# ------------------------------------------------------------
# 1.1 Determinar los objetivos del negocio
# ------------------------------------------------------------

st.header("1.1 Determinar los objetivos del negocio")

# ----------------------------
# Background
# ----------------------------

st.subheader("1.1.1 Background")

st.markdown("""
El mercado de vehículos usados en Costa Rica se caracteriza por una alta heterogeneidad en precios, marcas, modelos, antigüedad y kilometraje. 
Plataformas digitales como **Crautos.com** concentran una parte significativa de la oferta nacional y constituyen una fuente relevante de información pública sobre este mercado.

La determinación del precio suele realizarse de manera empírica, mediante comparaciones manuales o referencias subjetivas. 
Esto dificulta evaluar si un precio publicado es consistente con el comportamiento general del mercado.

El uso de técnicas de inteligencia artificial y aprendizaje automático permite transformar grandes volúmenes de datos en conocimiento estructurado 
que apoye la toma de decisiones relacionadas con la compra, venta y análisis del mercado automotriz.
""")

st.divider()

# ----------------------------
# Objetivos del negocio
# ----------------------------

st.subheader("1.1.2 Objetivos del negocio")

st.markdown("""
- **Estimar** de manera objetiva el precio de mercado de un vehículo usado en Costa Rica.
- **Analizar y segmentar** el mercado automotriz costarricense.
- **Identificar** los principales factores que influyen en la formación de precios.
""")

st.divider()

# ----------------------------
# Criterios de éxito
# ----------------------------

st.subheader("1.1.3 Criterios de éxito del negocio")

st.markdown("""
El proyecto se considerará exitoso si:

- Identifica factores relevantes en la determinación del precio.
- Genera segmentos interpretables y coherentes con perfiles reales de vehículos.
- Supera referencias triviales de predicción.
- Aporta interpretaciones útiles para la toma de decisiones.
""")

st.info(
    "El sistema desarrollado es una herramienta de apoyo y no un mecanismo determinístico de fijación de precios."
)

st.divider()
st.caption("TFG: Analítica del mercado de vehículos usados en Costa Rica | Metodología CRISP-DM")