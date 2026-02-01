import streamlit as st

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
with st.container(border=True):
    st.title("🧠 Feature Engineering")
    st.caption("Transición de CV Normal a CV Enriched")

st.write("")

# ------------------------------------------------------------
# Concepto general
# ------------------------------------------------------------
with st.container(border=True):
    st.header("📌 Objetivo del enriquecimiento")

    st.markdown("""
El dataset original proveniente del scraping (CV Normal) contenía variables técnicas y comerciales
directamente extraídas de los anuncios.

Sin embargo, para mejorar la capacidad explicativa y estructural de los modelos, se construyó una versión
**CV Enriched**, incorporando transformaciones, normalizaciones y nuevas variables derivadas.

El objetivo fue:
- Reducir inconsistencias textuales
- Incorporar variables de dominio
- Mejorar separabilidad estructural
- Facilitar modelado supervisado y no supervisado
""")

st.write("")

# ------------------------------------------------------------
# 1) Normalización de variables categóricas
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🔤 Normalización de variables categóricas")

    st.markdown("""
Se estandarizaron variables categóricas para evitar duplicados inconsistentes por mayúsculas,
acentos o variantes textuales.
""")

    with st.expander("Código: Normalización de marca, combustible y transmisión"):
        st.code("""
# Normalización de marca
df["marca_norm"] = (
    df["marca"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
)

# Normalización de combustible
df["combustible_norm"] = (
    df["combustible"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "NAN": np.nan,
        "ELÉCTRICO": "ELECTRICO",
        "ELECTRICO": "ELECTRICO",
        "HÍBRIDO": "HIBRIDO",
        "HIBRIDO": "HIBRIDO"
    })
)

# Normalización de transmisión
df["transmision_norm"] = (
    df["transmision"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "AUTOMÁTICA": "AUTOMATICA",
        "AUTOMATICA": "AUTOMATICA",
        "MANUAL": "MANUAL",
        "CVT": "CVT"
    })
)
""", language="python")

st.write("")

# ------------------------------------------------------------
# 2) Variables derivadas (ingeniería de dominio)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("⚙️ Variables derivadas")

    st.markdown("""
Se construyeron variables adicionales para capturar mejor la estructura del mercado.
""")

    st.markdown("""
**Variables clave agregadas:**
- `antiguedad`: años desde fabricación
- `marca_freq`: frecuencia relativa de la marca en el dataset
- `premium_flag`: indicador binario de marcas premium
""")

st.write("")

# ------------------------------------------------------------
# 3) Codificación de variables categóricas estratégicas
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧩 Codificación estructural")

    st.markdown("""
Se seleccionaron variables categóricas estratégicas y se aplicó One-Hot Encoding
con control de categorías desconocidas.
""")

    with st.expander("Código: OneHotEncoder"):
        st.code("""
cat_features = ["segmento_marca", "origen_marca", 
                "combustible_norm", "transmision_norm"]

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = encoder.fit_transform(df[cat_features])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(cat_features),
    index=df.index
)
""", language="python")

st.write("")

# ------------------------------------------------------------
# 4) Construcción del espacio final de modelado
# ------------------------------------------------------------
with st.container(border=True):
    st.header("📊 Construcción del espacio final (CV Enriched)")

    st.markdown("""
El CV Enriched se construyó combinando:

- Variables numéricas estructurales
- Variables derivadas
- Variables categóricas codificadas
""")

    with st.expander("Código: Construcción de X y escalado"):
        st.code("""
num_features = [
    "precio_usd",
    "kilometraje",
    "antiguedad",
    "cilindrada",
    "puertas",
    "marca_freq",
    "premium_flag"
]

X = pd.concat(
    [
        df[num_features],
        encoded_df
    ],
    axis=1
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
""", language="python")

st.write("")

# ------------------------------------------------------------
# Resultado conceptual
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🚀 Resultado del enriquecimiento")

    st.markdown("""
El paso de CV Normal a CV Enriched permitió:

- Reducir ruido textual
- Incorporar conocimiento de dominio
- Mejorar separabilidad estructural en clustering
- Aumentar capacidad predictiva en modelos supervisados
- Garantizar coherencia en el pipeline reproducible
""")

st.caption("TFG: Ingeniería de variables | Construcción del CV Enriched")