import streamlit as st

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# HERO
# ============================================================

with st.container(border=True):
    st.title("🧠 Feature Engineering")
    st.caption("Transformación del CV Normal al CV Enriched")

    st.markdown("""
El dataset original proveniente del scraping fue transformado
en un espacio estructurado (CV Enriched) mediante generación
de nuevas variables y criterios explícitos de ingeniería.
""")

st.write("")

# ============================================================
# DIFERENCIA ENTRE DATASETS
# ============================================================

with st.container(border=True):
    st.header("📊 Diferencia conceptual")

    st.markdown("""
**CV Normal**
- Variables directamente extraídas del anuncio
- Texto sin estandarización
- Sin variables estructurales derivadas

**CV Enriched**
- Variables normalizadas
- Variables derivadas
- Variables estructurales de dominio
- Espacio escalado para modelado
""")

st.write("")

# ============================================================
# GENERACIÓN DE VARIABLES DERIVADAS
# ============================================================

with st.container(border=True):
    st.header("⚙️ Generación de nuevas variables")

    # -------------------------
    # ANTIGÜEDAD
    # -------------------------
    st.subheader("1️⃣ Antigüedad")

    st.markdown("""
Se generó la variable `antiguedad` a partir del año del vehículo:

Antigüedad = Año_actual − Año_fabricación
""")

    st.markdown("""
**Criterio técnico:**

- El año absoluto no captura directamente depreciación.
- La antigüedad es una variable estructuralmente más informativa.
- Mejora gradiente temporal en regresión y separabilidad en clustering.
""")

    # -------------------------
    # MARCA_FREQ
    # -------------------------
    st.subheader("2️⃣ Frecuencia de marca (`marca_freq`)")

    st.markdown("""
Se calculó la frecuencia relativa de cada marca dentro del dataset.
""")

    st.markdown("""
**Criterio aplicado:**

- Las marcas con mayor presencia reflejan mayor penetración de mercado.
- Reduce efecto de alta cardinalidad.
- Introduce información estructural sin usar directamente el nombre de marca.
- Permite capturar popularidad como variable numérica.
""")

    # -------------------------
    # PREMIUM_FLAG
    # -------------------------
    st.subheader("3️⃣ Indicador Premium (`premium_flag`)")

    st.markdown("""
Se definió una variable binaria (0/1) para identificar marcas premium.
""")

    st.markdown("""
**Criterio de dominio:**

- El mercado automotriz costarricense presenta segmentación vertical.
- Las marcas premium siguen patrones de precio distintos.
- Facilita separación estructural en clustering jerárquico.
- Mejora desempeño en clasificación supervisada.
""")

st.write("")

# ============================================================
# NORMALIZACIÓN Y LIMPIEZA
# ============================================================

with st.container(border=True):
    st.header("🔤 Normalización y limpieza semántica")

    st.markdown("""
Se estandarizaron variables categóricas para evitar fragmentación
del espacio categórico por diferencias de formato.
""")

    st.markdown("""
**Problemas detectados en CV Normal:**
- Diferencias en mayúsculas/minúsculas
- Variaciones con y sin acentos
- Espacios inconsistentes
- Valores vacíos como strings

**Criterio aplicado:**
- Conversión a mayúsculas
- Eliminación de espacios
- Unificación de variantes equivalentes
""")

st.write("")

# ============================================================
# CODIFICACIÓN
# ============================================================

with st.container(border=True):
    st.header("🧩 Codificación estructural")

    st.markdown("""
Se aplicó One-Hot Encoding a variables estratégicas:

- segmento_marca
- origen_marca
- combustible_norm
- transmision_norm

**Criterio técnico:**
- No asumir orden artificial entre categorías
- Mantener interpretabilidad
- Permitir generalización con handle_unknown='ignore'
""")

st.write("")

# ============================================================
# ESCALADO
# ============================================================

with st.container(border=True):
    st.header("📏 Escalado del espacio")

    st.markdown("""
Se utilizó `StandardScaler` sobre variables numéricas.
""")

    st.markdown("""
**Criterio aplicado:**
- Evitar dominancia de variables de gran magnitud (ej. precio vs puertas)
- Mejorar estabilidad en clustering jerárquico (Ward)
- Facilitar convergencia en K-Means
""")

st.write("")

# ============================================================
# CRITERIOS GENERALES DE DISEÑO
# ============================================================

with st.container(border=True):
    st.header("🔎 Criterios generales de ingeniería")

    st.markdown("""
El enriquecimiento siguió cuatro principios:

1. Consistencia semántica  
2. Incorporación de conocimiento de dominio  
3. Mejora de separabilidad estructural  
4. Reproducibilidad del pipeline  

El CV Enriched no es simplemente un dataset limpio,
sino un espacio matemático diseñado para modelado.
""")

st.success("El enriquecimiento fue estructural y orientado a mejorar capacidad predictiva y segmentación.")