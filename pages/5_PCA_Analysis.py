import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import plotly.express as px

st.set_page_config(page_title="PCA Analysis", page_icon="🧩", layout="wide")

# ============================================================
# Replica EXACTA del notebook: 04_PCA_Analysis.ipynb
# PCA desde CR_Autos_Cleaned_enriched.csv
# Limpieza final + selección de variables + PCA + contribuciones
# ============================================================

# -----------------------------
# CONFIG (igual al notebook)
# -----------------------------
DATA_DIR = "data"
DEFAULT_CSV = "CR_Autos_Cleaned_enriched.csv"

COLS_DROP = [
    # comerciales / administrativos
    "impuestos_pagados",
    "precio_negociable",
    "recibe_vehiculo",
    # para evitar mezcla CRC/USD (usas CRC como principal)
    "precio_usd",
    # alta cardinalidad / identificadores (no para PCA)
    "marca",
    "modelo",
]

NUMERIC_CANDIDATES = [
    "precio_crc",
    "kilometraje",
    "antiguedad",
    "cilindrada",
    "puertas",
    "pasajeros",
    "participacion_mercado",
]

CATEGORICAL_CORE = [
    "estilo",
    "combustible",
    "transmision",
    "segmento_marca",
    "origen_marca",
]

CATEGORICAL_OPTIONAL = [
    "estado",
    "provincia",
    "color_exterior",
    "color_interior",
]


# -----------------------------
# HELPERS (mismo enfoque)
# -----------------------------
def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
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
        cat_names = list(ohe.get_feature_names_out(categorical_features))
        feature_names += cat_names
    return feature_names


def top_features_with_sign(loadings_df: pd.DataFrame, pc="PC1", n=12):
    s = loadings_df[pc].sort_values(key=lambda x: x.abs(), ascending=False).head(n)
    return s.to_frame(name="loading").reset_index().rename(columns={"index": "feature"})


# ============================================================
# UI
# ============================================================
st.title("🧩 PCA Analysis (replicando el Jupyter)")
st.caption("Lee el CSV desde `./data/`, aplica el mismo flujo del notebook `04_PCA_Analysis.ipynb` y muestra PCA + loadings + 3D.")

with st.container(border=True):
    st.subheader("1) Entrada")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        csv_name = st.text_input("CSV dentro de /data", value=DEFAULT_CSV)

    with col2:
        incluir_opcionales = st.checkbox("INCLUIR_OPCIONALES", value=False)

    with col3:
        n_components = st.number_input("n_components", min_value=2, max_value=6, value=3, step=1)

    path = os.path.join(DATA_DIR, csv_name)
    st.write("Ruta esperada:", f"`{path}`")

    if not os.path.exists(path):
        st.error(f"No encontré `{path}`. Pon el CSV en el folder `data/`.")
        st.stop()

# ------------------------------------------------------------
# 1) Cargar dataset (igual notebook)
# ------------------------------------------------------------
df = pd.read_csv(path)

with st.container(border=True):
    st.subheader("2) Carga y limpieza (igual al notebook)")

    st.write("Shape original:", df.shape)

    # 2) Eliminar duplicados exactos
    dup = int(df.duplicated().sum())
    st.write("Duplicados exactos detectados:", dup)

    df = df.drop_duplicates()
    st.write("Filas después de eliminar duplicados:", df.shape[0])

    # 3) Drop columnas no deseadas
    df = df.drop(columns=[c for c in COLS_DROP if c in df.columns], errors="ignore")
    st.write("Shape después de drop columnas:", df.shape)

    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------
# 4) Definir features para PCA (igual notebook)
# ------------------------------------------------------------
categorical_candidates = CATEGORICAL_CORE + (CATEGORICAL_OPTIONAL if incluir_opcionales else [])
numeric_features = [c for c in NUMERIC_CANDIDATES if c in df.columns]
categorical_features = [c for c in categorical_candidates if c in df.columns]
selected_features = numeric_features + categorical_features

with st.container(border=True):
    st.subheader("3) Variables seleccionadas para PCA")
    st.write("Numéricas:", numeric_features)
    st.write("Categóricas:", categorical_features)
    st.write("Total:", len(selected_features))

    if len(selected_features) == 0:
        st.error("No se detectó ninguna variable candidata en el CSV.")
        st.stop()

df_model = df[selected_features].copy()

# ------------------------------------------------------------
# 5) Tratamiento de faltantes críticos (igual notebook)
# ------------------------------------------------------------
critical_cols = [c for c in ["precio_crc", "kilometraje", "antiguedad"] if c in df_model.columns]

with st.container(border=True):
    st.subheader("4) Tratamiento de faltantes (críticos)")
    st.write("Variables críticas usadas para dropna:", critical_cols)

    before = df_model.shape[0]
    df_model = df_model.dropna(subset=critical_cols)
    after = df_model.shape[0]

    st.write("Filas antes:", before)
    st.write("Filas después:", after)
    st.write("Filas removidas por faltantes críticos:", before - after)

# ------------------------------------------------------------
# 6) Pipeline de preprocesamiento + PCA (igual notebook)
# ------------------------------------------------------------
preprocessor = build_preprocessor(numeric_features, categorical_features)
X = preprocessor.fit_transform(df_model)

# PCA necesita denso por OneHot
X_dense = X.toarray() if hasattr(X, "toarray") else X

with st.container(border=True):
    st.subheader("5) Preprocesamiento")
    st.write("Dimensión tras preprocesamiento (One-Hot + escala):", X_dense.shape)

pca = PCA(n_components=int(n_components), random_state=42)
X_pca = pca.fit_transform(X_dense)

explained_var = pca.explained_variance_ratio_

pc_cols = [f"PC{i}" for i in range(1, int(n_components) + 1)]
df_pca = pd.DataFrame(X_pca, columns=pc_cols, index=df_model.index)

feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=pc_cols
)

# ------------------------------------------------------------
# 7) Varianza explicada + head (igual notebook)
# ------------------------------------------------------------
with st.container(border=True):
    st.subheader("6) PCA - Varianza explicada")
    var_tbl = pd.DataFrame({
        "Componente": pc_cols,
        "Varianza explicada": explained_var,
        "Varianza explicada (%)": (explained_var * 100).round(2),
    })
    var_tbl["Varianza acumulada (%)"] = var_tbl["Varianza explicada (%)"].cumsum().round(2)
    st.dataframe(var_tbl, use_container_width=True)
    st.write(f"Varianza acumulada (PC1–PC{int(n_components)}): {explained_var.sum():.4f}")

with st.container(border=True):
    st.subheader("7) Head PCA")
    st.dataframe(df_pca.head(10), use_container_width=True)

# ------------------------------------------------------------
# 8) TOP FEATURES por componente (por |loading|) (igual notebook)
# ------------------------------------------------------------
with st.container(border=True):
    st.subheader("8) Top features por componente (|loading|)")
    top_n = st.slider("TOP_N", min_value=5, max_value=30, value=15, step=1)

    tabs = st.tabs(pc_cols)
    for pc, tab in zip(pc_cols, tabs):
        with tab:
            st.markdown(f"**{pc} (Top {top_n} por |loading|)**")
            s = loadings[pc].abs().sort_values(ascending=False).head(top_n)
            st.dataframe(s.reset_index().rename(columns={"index": "feature", pc: "abs_loading"}),
                         use_container_width=True)

            st.markdown("**Top con signo (para interpretación)**")
            st.dataframe(top_features_with_sign(loadings, pc=pc, n=min(12, top_n)),
                         use_container_width=True)

# ------------------------------------------------------------
# 9) Visualización 3D (igual notebook, si hay PC1-3)
# ------------------------------------------------------------
with st.container(border=True):
    st.subheader("9) PCA 3D – Compresión de datos (Preparación)")
    if int(n_components) < 3:
        st.info("Para scatter 3D necesitas al menos 3 componentes.")
    else:
        fig = px.scatter_3d(
            df_pca,
            x="PC1", y="PC2", z="PC3",
            opacity=0.6,
            title="PCA 3D – Compresión de datos (Preparación)"
        )
        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(fig, use_container_width=True)