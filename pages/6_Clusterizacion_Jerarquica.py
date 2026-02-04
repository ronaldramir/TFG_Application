import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

st.set_page_config(page_title="Clustering Jerárquico (Ward)", page_icon="🌳", layout="wide")

# ============================================================
# Configuración (alineada a tu flujo con Cleaned Enriched)
# ============================================================

DEFAULT_DROP_COLS = [
    "impuestos_pagados",
    "precio_negociable",
    "recibe_vehiculo",
    "precio_usd",
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


def ward_linkage(X_dense):
    # Ward trabaja con distancia euclidiana en el espacio transformado
    return linkage(X_dense, method="ward")


# ============================================================
# UI
# ============================================================

st.title("🌳 Clustering Jerárquico (Ward)")
st.caption("Carga `CR_Autos_Cleaned_enriched.csv`, aplica el mismo preprocesamiento que PCA y ejecuta Ward + dendrograma + corte K.")

with st.container(border=True):
    st.subheader("1) Entrada de datos")
    up = st.file_uploader("Subir CSV (recomendado: CR_Autos_Cleaned_enriched.csv)", type=["csv"])

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        K = st.number_input("K (clusters)", min_value=2, max_value=10, value=3, step=1)
    with col2:
        include_optional = st.toggle("Incluir categóricas opcionales", value=False)
    with col3:
        drop_dups = st.toggle("Eliminar duplicados exactos", value=True)
    with col4:
        st.caption("Opcionales: `estado`, `provincia`, `color_exterior`, `color_interior` (pueden inflar One-Hot).")

    st.divider()

    st.subheader("2) Limpieza final")
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

    st.divider()
    st.subheader("3) Dendrograma")
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        truncate_mode = st.selectbox("truncate_mode", ["lastp", "level", "none"], index=0)
    with colB:
        p = st.number_input("p (si truncado)", min_value=10, max_value=200, value=40, step=5)
    with colC:
        st.caption("`lastp` muestra los últimos p clusters fusionados. Útil para datasets grandes.")

    st.divider()
    run_btn = st.button("Ejecutar Ward + dendrograma + asignación", type="primary", use_container_width=True)

if up is None:
    st.info("Sube el archivo para ejecutar el clustering jerárquico.")
    st.stop()

# ============================================================
# Cargar y preparar datos
# ============================================================

df = pd.read_csv(up)
orig_shape = df.shape

if drop_dups:
    df = df.drop_duplicates()

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

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

with st.container(border=True):
    st.subheader("Resumen del dataset")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Shape original", f"{orig_shape[0]} x {orig_shape[1]}")
    with c2:
        st.metric("Tras drops", f"{df.shape[0]} x {df.shape[1]}")
    with c3:
        st.metric("Para clustering", f"{df_model.shape[0]} x {df_model.shape[1]}")
    with c4:
        st.metric("Filas removidas (NaN críticos)", before - after)

    st.markdown("**Features usadas:**")
    st.write({"numéricas": numeric_features, "categóricas": categorical_features, "total": len(selected_features)})

if not run_btn:
    st.stop()

# ============================================================
# Preprocesamiento + Ward
# ============================================================

preprocessor = build_preprocessor(numeric_features, categorical_features)
X = preprocessor.fit_transform(df_model)
X_dense = X.toarray() if hasattr(X, "toarray") else X

with st.container(border=True):
    st.subheader("Preprocesamiento")
    st.metric("Dimensión tras One-Hot + escala", f"{X_dense.shape[0]} x {X_dense.shape[1]}")
    st.caption("Ward utiliza distancias euclidianas sobre este espacio transformado.")

# Linkage
with st.spinner("Calculando linkage (Ward)..."):
    Z = ward_linkage(X_dense)

# Dendrograma
with st.container(border=True):
    st.subheader("Dendrograma (Ward)")
    fig, ax = plt.subplots(figsize=(14, 6))
    kwargs = dict(
        Z=Z,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True
    )

    if truncate_mode != "none":
        dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=int(p),
            ax=ax,
            **{k: v for k, v in kwargs.items() if k != "Z"}
        )
    else:
        dendrogram(
            Z,
            ax=ax,
            **{k: v for k, v in kwargs.items() if k != "Z"}
        )

    ax.set_title(f"Dendrograma (Ward) — truncado para visualización (K={int(K)} sugerido)")
    ax.set_xlabel("Clusters fusionados")
    ax.set_ylabel("Distancia (Ward)")
    st.pyplot(fig, clear_figure=True)

# Corte en K
labels_hc = fcluster(Z, t=int(K), criterion="maxclust") - 1

# Crear salida alineada por índice
df_out = df_model.copy()
df_out["cluster_id_hc"] = labels_hc

with st.container(border=True):
    st.subheader("Asignación de clusters")
    vc = df_out["cluster_id_hc"].value_counts().sort_index()
    st.write("**Tamaño de clusters:**")
    st.dataframe(vc.rename("conteo").to_frame(), use_container_width=True)

    st.markdown("**Vista previa:**")
    st.dataframe(df_out.head(25), use_container_width=True)

# Descarga
with st.container(border=True):
    st.subheader("Descarga")
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar dataset con clusters (Ward)",
        data=csv_bytes,
        file_name=f"Unsupervised_Learning_HC_WARD_K{int(K)}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.success("Listo: dendrograma generado y clusters asignados.")
