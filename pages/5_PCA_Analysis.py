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
# Configuración base (según tu notebook PCA_Analysis)
# ============================================================

# Columnas a excluir (ruido comercial / no estructural / alta cardinalidad directa)
DEFAULT_DROP_COLS = [
    "impuestos_pagados",
    "precio_negociable",
    "recibe_vehiculo",
    "precio_usd",
    "marca",
    "modelo",
]

# Variables candidatas (de tu CSV completo)
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


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = list(numeric_features)
    if len(categorical_features) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out(categorical_features))
        feature_names += cat_names
    return feature_names


def run_pca(df_model, numeric_features, categorical_features, n_components=3, random_state=42):
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X = preprocessor.fit_transform(df_model)

    X_dense = X.toarray() if hasattr(X, "toarray") else X

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_dense)

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i}" for i in range(1, n_components + 1)]
    )

    return X_pca, pca, loadings, X_dense.shape


# ============================================================
# UI
# ============================================================

st.title("🧩 PCA Analysis (desde `CR_Autos_Cleaned_enriched.csv`)")
st.caption("Carga el dataset enriquecido, aplica selección final de variables y ejecuta PCA para compresión + interpretación.")

with st.container(border=True):
    st.subheader("1) Entrada de datos")
    st.markdown("Sube `CR_Autos_Cleaned_enriched.csv` (o un CSV con las mismas columnas).")

    up = st.file_uploader("Subir CSV", type=["csv"])

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        n_components = st.number_input("Componentes PCA", min_value=2, max_value=10, value=3, step=1)
    with col2:
        include_optional = st.toggle("Incluir categóricas opcionales", value=False)
    with col3:
        drop_dups = st.toggle("Eliminar duplicados exactos", value=True)
    with col4:
        st.caption("Opcionales: `estado`, `provincia`, `color_exterior`, `color_interior` (pueden inflar One-Hot).")

    st.divider()

    st.subheader("2) Limpieza final (lo mínimo necesario)")
    drop_cols = st.multiselect(
        "Columnas a eliminar (recomendado)",
        options=sorted(list(set(DEFAULT_DROP_COLS))),
        default=DEFAULT_DROP_COLS
    )

    critical_default = ["precio_crc", "kilometraje", "antiguedad"]
    critical_cols = st.multiselect(
        "Columnas críticas para eliminar filas con faltantes (dropna)",
        options=sorted(list(set(NUMERIC_CANDIDATES))),
        default=[c for c in critical_default if c in NUMERIC_CANDIDATES]
    )

    run_btn = st.button("Ejecutar PCA", type="primary", use_container_width=True)

# ============================================================
# Procesamiento
# ============================================================

if up is None:
    st.info("Sube el archivo para poder ejecutar el PCA.")
    st.stop()

df = pd.read_csv(up)
orig_shape = df.shape

if drop_dups:
    df = df.drop_duplicates()

# Drop columnas
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Selección de features
categorical_candidates = CATEGORICAL_CORE + (CATEGORICAL_OPTIONAL if include_optional else [])
numeric_features = [c for c in NUMERIC_CANDIDATES if c in df.columns]
categorical_features = [c for c in categorical_candidates if c in df.columns]
selected_features = numeric_features + categorical_features

df_model = df[selected_features].copy()

# Drop NaN críticos
crit = [c for c in critical_cols if c in df_model.columns]
before = df_model.shape[0]
if len(crit) > 0:
    df_model = df_model.dropna(subset=crit)
after = df_model.shape[0]

# Mostrar resumen
with st.container(border=True):
    st.subheader("Resumen del dataset para PCA")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Shape original", f"{orig_shape[0]} x {orig_shape[1]}")
    with c2:
        st.metric("Tras drops", f"{df.shape[0]} x {df.shape[1]}")
    with c3:
        st.metric("Para PCA", f"{df_model.shape[0]} x {df_model.shape[1]}")
    with c4:
        st.metric("Filas removidas (NaN críticos)", before - after)

    st.markdown("**Features usadas:**")
    st.write({"numéricas": numeric_features, "categóricas": categorical_features, "total": len(selected_features)})

if not run_btn:
    st.stop()

# ============================================================
# PCA
# ============================================================

X_pca, pca_obj, loadings, dense_shape = run_pca(
    df_model=df_model,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    n_components=int(n_components),
    random_state=42
)

explained = pca_obj.explained_variance_ratio_
cum_explained = explained.cumsum()

with st.container(border=True):
    st.subheader("Resultados PCA")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dimensión tras preprocesamiento", f"{dense_shape[0]} x {dense_shape[1]}")
    with c2:
        st.metric("Varianza acumulada (1..k)", f"{cum_explained[-1]:.4f}")
    with c3:
        st.metric("k (componentes)", int(n_components))

    # Tabla varianza
    var_df = pd.DataFrame({
        "Componente": [f"PC{i}" for i in range(1, int(n_components) + 1)],
        "Varianza explicada": explained,
        "Varianza acumulada": cum_explained
    })
    st.dataframe(var_df, use_container_width=True)

    # Gráfico varianza
    fig_var = px.bar(
        var_df,
        x="Componente",
        y="Varianza explicada",
        title="Varianza explicada por componente"
    )
    st.plotly_chart(fig_var, use_container_width=True)

# ============================================================
# Loadings (contribución de variables)
# ============================================================

with st.container(border=True):
    st.subheader("Variables que más contribuyen por componente (|loading|)")

    top_n = st.slider("Top N", min_value=5, max_value=30, value=15, step=1)
    tabs = st.tabs([f"PC{i}" for i in range(1, int(n_components) + 1)])

    for i, tab in enumerate(tabs, start=1):
        pc = f"PC{i}"
        with tab:
            top_abs = loadings[pc].abs().sort_values(ascending=False).head(top_n)
            st.dataframe(top_abs.rename("abs_loading").to_frame(), use_container_width=True)

            st.markdown("**Top con signo (útil para interpretación):**")
            top_signed = loadings[pc].sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
            st.dataframe(top_signed.rename("loading").to_frame(), use_container_width=True)

# ============================================================
# Visualización PCA (2D y 3D)
# ============================================================

pca_cols = [f"PC{i}" for i in range(1, int(n_components) + 1)]
df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df_model.index)

with st.container(border=True):
    st.subheader("Visualización")
    st.markdown("Scatter 2D (PC1 vs PC2) y 3D (PC1, PC2, PC3 si aplica).")

    if int(n_components) >= 2:
        fig2d = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            opacity=0.6,
            title="PCA 2D (PC1 vs PC2)"
        )
        st.plotly_chart(fig2d, use_container_width=True)

    if int(n_components) >= 3:
        fig3d = px.scatter_3d(
            df_pca,
            x="PC1",
            y="PC2",
            z="PC3",
            opacity=0.6,
            title="PCA 3D (PC1, PC2, PC3)"
        )
        fig3d.update_traces(marker=dict(size=3))
        st.plotly_chart(fig3d, use_container_width=True)

# ============================================================
# Descargas
# ============================================================

with st.container(border=True):
    st.subheader("Descargas")
    st.markdown("Puedes descargar el dataset PCA y los loadings para documentar resultados.")

    csv_pca = df_pca.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar PCA (componentes)",
        data=csv_pca,
        file_name="pca_components.csv",
        mime="text/csv",
        use_container_width=True
    )

    csv_loadings = loadings.to_csv().encode("utf-8")
    st.download_button(
        "Descargar loadings (contribuciones)",
        data=csv_loadings,
        file_name="pca_loadings.csv",
        mime="text/csv",
        use_container_width=True
    )
