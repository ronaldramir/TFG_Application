import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import plotly.express as px

# ============================================================
# Configuración de página
# ============================================================
st.set_page_config(
    page_title="PCA | Compresión de datos",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
with st.container(border=True):
    st.title("🧠 PCA: Compresión y estructura del mercado")
    st.caption("Versión Streamlit del notebook: selección de variables → preprocesamiento → PCA 3D → contribuciones (loadings)")

st.write("")

# ============================================================
# Helpers
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file, default_path: str):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(default_path)

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = list(numeric_features)
    if len(categorical_features) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feature_names += list(ohe.get_feature_names_out(categorical_features))
    return feature_names

# ============================================================
# Sidebar: Controles
# ============================================================
with st.sidebar:
    st.header("⚙️ Controles")
    uploaded = st.file_uploader("Cargar CSV (opcional)", type=["csv"])

    default_path = st.text_input(
        "Ruta CSV por defecto (si no subes archivo)",
        value="CR_Autos_Cleaned_enriched.csv"
    )

    st.divider()

    INCLUIR_OPCIONALES = st.toggle(
        "Incluir categóricas opcionales (alta cardinalidad)",
        value=False,
        help="Incluye: estado, provincia, color_exterior, color_interior. Puede aumentar muchísimo el One-Hot."
    )

    TOP_N = st.slider("Top N features por componente (|loading|)", min_value=5, max_value=30, value=15)

    st.divider()
    show_sample = st.checkbox("Mostrar muestra del dataset", value=True)
    show_selected = st.checkbox("Mostrar variables seleccionadas", value=True)
    show_loadings_table = st.checkbox("Mostrar tabla completa de loadings", value=False)
    show_code = st.checkbox("Mostrar snippet del código base (Jupyter)", value=False)

# ============================================================
# 1) Cargar dataset
# ============================================================
with st.container(border=True):
    st.header("📥 Carga de datos")

    try:
        df = load_data(uploaded, default_path)
        st.success(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]:,} columnas")
    except Exception as e:
        st.error("No se pudo cargar el CSV. Revisa la ruta o sube el archivo.")
        st.exception(e)
        st.stop()

    if show_sample:
        st.subheader("Vista rápida")
        st.dataframe(df.head(25), use_container_width=True)

st.write("")

# ============================================================
# 2) Limpieza mínima del notebook (duplicados + drops)
# ============================================================
with st.container(border=True):
    st.header("🧹 Limpieza mínima (como en el notebook)")

    dup = int(df.duplicated().sum())
    st.write(f"Duplicados exactos detectados: **{dup:,}**")
    df = df.drop_duplicates()

    cols_drop = [
        "impuestos_pagados",
        "precio_negociable",
        "recibe_vehiculo",
        "precio_usd",
        "marca",
        "modelo",
    ]
    to_drop = [c for c in cols_drop if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    st.write(f"Columnas eliminadas: **{len(to_drop)}** → {to_drop if to_drop else 'Ninguna (no estaban en el CSV)'}")
    st.write(f"Shape después de limpieza: **{df.shape[0]:,} filas × {df.shape[1]:,} columnas**")

st.write("")

# ============================================================
# 3) Selección de variables para PCA (idéntico al notebook)
# ============================================================
numeric_candidates = [
    "precio_crc",
    "kilometraje",
    "antiguedad",
    "cilindrada",
    "puertas",
    "pasajeros",
    "participacion_mercado",
]

categorical_core = [
    "estilo",
    "combustible",
    "transmision",
    "segmento_marca",
    "origen_marca",
]

categorical_optional = [
    "estado",
    "provincia",
    "color_exterior",
    "color_interior",
]

categorical_candidates = categorical_core + (categorical_optional if INCLUIR_OPCIONALES else [])

numeric_features = [c for c in numeric_candidates if c in df.columns]
categorical_features = [c for c in categorical_candidates if c in df.columns]
selected_features = numeric_features + categorical_features

with st.container(border=True):
    st.header("🧩 Variables seleccionadas para PCA")

    if len(selected_features) == 0:
        st.error("No se encontró ninguna variable candidata en el CSV. Revisa nombres/columnas.")
        st.stop()

    if show_selected:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numéricas**")
            st.write(numeric_features if numeric_features else "Ninguna")
        with c2:
            st.markdown("**Categóricas**")
            st.write(categorical_features if categorical_features else "Ninguna")

    st.write(f"Total variables usadas (antes de One-Hot): **{len(selected_features)}**")

    df_model = df[selected_features].copy()

st.write("")

# ============================================================
# 4) Tratamiento de faltantes críticos (dropna) + imputación
# ============================================================
with st.container(border=True):
    st.header("🩹 Faltantes (críticos + pipeline)")

    critical_cols = [c for c in ["precio_crc", "kilometraje", "antiguedad"] if c in df_model.columns]
    if len(critical_cols) == 0:
        st.warning("No se encontraron columnas críticas (precio_crc/kilometraje/antiguedad). Se omite dropna por críticos.")
    else:
        before = df_model.shape[0]
        df_model = df_model.dropna(subset=critical_cols)
        after = df_model.shape[0]
        st.write(f"Dropna por críticos: {critical_cols}")
        st.write(f"Filas antes: **{before:,}** | después: **{after:,}** | removidas: **{before-after:,}**")

st.write("")

# ============================================================
# 5) Preprocesamiento + PCA 3D
# ============================================================
with st.container(border=True):
    st.header("🧠 PCA 3D (fit)")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    with st.spinner("Preprocesando (One-Hot + escala) y entrenando PCA..."):
        X = preprocessor.fit_transform(df_model)

        # PCA necesita denso (por OneHot)
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(X_dense)

    explained_var = pca.explained_variance_ratio_
    var_acum = float(np.sum(explained_var))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PC1", f"{explained_var[0]*100:.2f}%")
    c2.metric("PC2", f"{explained_var[1]*100:.2f}%")
    c3.metric("PC3", f"{explained_var[2]*100:.2f}%")
    c4.metric("Acumulada PC1–PC3", f"{var_acum*100:.2f}%")

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"], index=df_model.index)

    st.subheader("Scatter 3D interactivo")
    color_col = None
    # Si tienes cluster en el dataset, lo usamos para colorear automáticamente
    for candidate in ["cluster", "cluster_id_hc", "cluster_id", "cluster_k3", "cluster_k4"]:
        if candidate in df.columns:
            color_col = candidate
            break

    if color_col is not None:
        tmp = df_pca.join(df[[color_col]], how="left")
        fig = px.scatter_3d(
            tmp, x="PC1", y="PC2", z="PC3",
            color=color_col,
            opacity=0.65,
            title=f"PCA 3D – Compresión de datos (color: {color_col})"
        )
    else:
        fig = px.scatter_3d(
            df_pca, x="PC1", y="PC2", z="PC3",
            opacity=0.65,
            title="PCA 3D – Compresión de datos (Preparación)"
        )

    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True)

st.write("")

# ============================================================
# 6) Loadings: contribución de variables
# ============================================================
with st.container(border=True):
    st.header("📌 Contribución de variables (loadings)")

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PC1", "PC2", "PC3"]
    )

    tabs = st.tabs(["Top por componente", "Top con signo", "Tabla (opcional)"])

    with tabs[0]:
        for pc in ["PC1", "PC2", "PC3"]:
            st.subheader(f"{pc} (Top {TOP_N} por |loading|)")
            top_abs = loadings[pc].abs().sort_values(ascending=False).head(TOP_N)
            st.dataframe(top_abs.to_frame(name="|loading|"), use_container_width=True)

    with tabs[1]:
        pc_pick = st.selectbox("Componente", ["PC1", "PC2", "PC3"], index=0)
        n_pick = st.slider("Cantidad", 5, 30, 12)
        s = loadings[pc_pick].sort_values(key=lambda x: x.abs(), ascending=False).head(n_pick)
        st.dataframe(s.to_frame(name="loading (con signo)"), use_container_width=True)

    with tabs[2]:
        if show_loadings_table:
            st.dataframe(loadings, use_container_width=True, height=420)
        else:
            st.info("Activa en el sidebar: 'Mostrar tabla completa de loadings' si de verdad quieres ver todo (spoiler: es largo).")

st.write("")

# ============================================================
# 7) Conclusiones (del notebook)
# ============================================================
with st.container(border=True):
    st.header("🧾 Conclusiones (resumen)")
    st.markdown(r"""
## Conclusiones del Análisis PCA (Preparación de los Datos)

Con el objetivo de comprender la estructura subyacente del mercado de vehículos usados y reducir la complejidad del espacio de variables, se aplicó un Análisis de Componentes Principales (PCA) sobre el dataset `CR_Autos_Cleaned_enriched.csv`, una vez realizada la selección de variables, eliminación de duplicados y tratamiento de valores faltantes críticos.

### Varianza explicada
El PCA tridimensional explica aproximadamente el **48.7% de la varianza total** del conjunto de datos. Este valor es consistente con un dataset que combina variables numéricas y categóricas (one-hot encoded), y resulta suficiente para propósitos exploratorios, visualización y análisis estructural del mercado.

---

### Interpretación de los componentes principales

### Componente Principal 1 (PC1): Eje de depreciación y valor económico
El primer componente principal está dominado por variables como:

- **Antigüedad**
- **Kilometraje**
- **Precio (CRC)**, con signo opuesto
- **Participación de mercado**
- **Segmento de marca generalista**

Este eje representa una clara oposición entre vehículos **más usados, antiguos y comunes en el mercado**, frente a vehículos **más nuevos, menos usados y de mayor valor económico**.  
PC1 puede interpretarse como un eje de **depreciación y posicionamiento económico**, siendo el principal factor diferenciador del mercado de vehículos usados.

---

### Componente Principal 2 (PC2): Eje de configuración funcional y capacidad
El segundo componente está principalmente influenciado por:

- **Número de pasajeros**
- **Número de puertas**
- **Estilo del vehículo (SUV, AWD)**
- **Participación de mercado**
- **Segmento de marca**

Este componente no está dominado por el precio, sino por la **configuración física y funcional del vehículo**, diferenciando autos compactos de vehículos familiares o utilitarios.  
PC2 puede interpretarse como un eje de **uso práctico y capacidad**, asociado al tipo de necesidad que cubre el vehículo.

---

### Componente Principal 3 (PC3): Eje de perfil mecánico y tecnológico
El tercer componente presenta mayor influencia de:

- **Cilindrada**
- **Tipo de combustible**
- **Estilo del vehículo (SUV, Pick-Up)**
- **Kilometraje y antigüedad**
- **Origen de la marca**

Este eje captura diferencias relacionadas con el **perfil mecánico y tecnológico** del vehículo, distinguiendo entre autos de mayor potencia o capacidad mecánica y vehículos más convencionales.  
PC3 aporta una dimensión adicional que complementa los dos primeros ejes, sin dominar la estructura global.

---

### Evaluación general del PCA
El análisis confirma que:

- El mercado de vehículos usados se estructura principalmente alrededor de:
  - **Uso y depreciación**
  - **Configuración funcional**
  - **Perfil mecánico**
- Las variables derivadas de marca (`segmento_marca`, `origen_marca`, `participacion_mercado`) aportan información relevante sin dominar el análisis.
- La exclusión de variables como `marca`, `modelo`, colores y ubicación geográfica evita sesgos por alta cardinalidad y permite que el PCA capture relaciones estructurales reales.

En conjunto, el PCA cumple adecuadamente su rol como **herramienta exploratoria y de compresión**, sirviendo de base para análisis posteriores como la segmentación por clustering y la interpretación de perfiles de mercado.
""")

st.write("")

# ============================================================
# 8) Código base (opcional)
# ============================================================
if show_code:
    with st.container(border=True):
        st.header("🧩 Código base del notebook (snippet)")
        st.code("""
# (extracto) Pipeline + PCA 3D (notebook)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical_features),
    ],
    remainder="drop"
)

X = preprocessor.fit_transform(df_model)
X_dense = X.toarray() if hasattr(X, "toarray") else X

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_dense)
""", language="python")
