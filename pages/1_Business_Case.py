import streamlit as st

st.set_page_config(
    page_title="Business Case | CRISP-DM",
    page_icon="📌",
    layout="centered"
)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
with st.container(border=True):
    st.title("📌 Comprensión del Negocio")
    st.caption("Contexto, objetivos y criterios de éxito del proyecto.")
    st.markdown(
        "**Metodología:** CRISP-DM  \n"
        "**Dominio:** Mercado de vehículos usados en Costa Rica"
    )

st.write("")  # espacio

# ------------------------------------------------------------
# Background
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧭 Background")
    st.markdown("""
El mercado de vehículos usados en Costa Rica se caracteriza por una alta heterogeneidad en precios, marcas, modelos, antigüedad y kilometraje. 
Plataformas digitales como **Crautos.com** concentran una parte significativa de la oferta nacional y constituyen una fuente relevante de información pública sobre este mercado.

La determinación del precio suele realizarse de manera empírica, mediante comparaciones manuales o referencias subjetivas. Esto dificulta evaluar si un precio publicado es consistente con el comportamiento general del mercado.

El uso de técnicas de inteligencia artificial y aprendizaje automático permite transformar grandes volúmenes de datos en conocimiento estructurado que apoye la toma de decisiones relacionadas con la compra, venta y análisis del mercado automotriz.
""")

st.write("")

# ------------------------------------------------------------
# Objetivos del negocio
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🎯 Objetivos del negocio")
    st.markdown("""
- Estimar de manera objetiva el precio de mercado de un vehículo usado en Costa Rica.
- Analizar y segmentar el mercado automotriz costarricense.
- Identificar los principales factores que influyen en la formación de precios.
""")

st.write("")

# ------------------------------------------------------------
# Criterios de éxito
# ------------------------------------------------------------
with st.container(border=True):
    st.header("✅ Criterios de éxito")
    st.markdown("""
El proyecto se considerará exitoso si:

- Identifica factores relevantes en la determinación del precio.
- Genera segmentos interpretables y coherentes con perfiles reales de vehículos.
- Aporta interpretaciones útiles para la toma de decisiones.
""")
    st.info("El sistema desarrollado es una herramienta de apoyo y no un mecanismo determinístico de fijación de precios.")

st.write("")

# ------------------------------------------------------------
# Inventario de recursos
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧰 Inventario de recursos")
    st.markdown("""
- Dataset completo extraído desde Crautos.com.
- Python como lenguaje principal.
- Librerías: pandas, numpy, scikit-learn.
- Infraestructura computacional personal.
- Aplicación web en Streamlit para despliegue del demo.
""")

st.write("")

# ------------------------------------------------------------
# Requisitos, supuestos y restricciones
# ------------------------------------------------------------
with st.container(border=True):
    st.header("📌 Requisitos, supuestos y restricciones")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Requisitos")
        st.markdown("""
- Uso exclusivo de datos públicos disponibles en Crautos.com.
- Implementación de modelos directamente en Python.
- Aplicación explícita de la metodología CRISP-DM.
""")
    with c2:
        st.subheader("Supuestos")
        st.markdown("""
- El precio publicado es una aproximación razonable al valor de mercado.
- Las variables disponibles contienen información suficiente para modelar el precio.
""")
    with c3:
        st.subheader("Restricciones")
        st.markdown("""
- No se dispone del precio final de venta.
- El análisis se limita al período cubierto por la extracción.
- La calidad depende de la exactitud de los anuncios.
""")

st.write("")

# ------------------------------------------------------------
# Riesgos y contingencias
# ------------------------------------------------------------
with st.container(border=True):
    st.header("⚠️ Riesgos y contingencias")
    st.markdown("""
- Presencia de valores atípicos (outliers).
- Registros incompletos o inconsistentes.
- Alta cardinalidad en variables categóricas.
""")
    st.warning("Se aplican técnicas de limpieza, transformación y validación cruzada para mitigar estos riesgos.")

st.write("")
st.caption("TFG: Analítica del mercado de vehículos usados en Costa Rica | Metodología CRISP-DM")