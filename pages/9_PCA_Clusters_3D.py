import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import plotly.express as px

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="PCA + Clusters (3D)", page_icon="🧠", layout="wide")

with st.container(border=True):
    st.title("🧠 PCA 3D con colores por cluster (Ward)")
    st.caption("Carga → preproceso (escala + one-hot) → (opcional) Ward K → PCA 3D/2D coloreado por cluster")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Controles")

    uploaded = st.file_uploader("Cargar CSV (opcional)", type=["csv"])
    default_path = st.text_input(
        "Ruta CSV por defecto",
        value="data/CR_Autos_Cleaned_enriched.csv"
    )

    st.divider()
    st.subheader("Cluster")
    K = st.slider("K (Ward)", 2, 8, 3)
    compute_clusters = st.toggle(
        "Calcular Ward si no existe cluster_id_hc",
        value=True,
        help="Si el CSV ya trae cluster_id_hc, se usa y no se recalcula."
    )
    use_sampling = st.toggle(
        "Muestreo para Ward (recomendado con >5k filas)",
        value=True,
        help="Ward escala mal. Esto evita que el cálculo sea una tragedia."
    )
    max_rows = st.slider("Máximo filas para Ward (si muestreo)", 500, 8000, 2500, step=250)

    st.divider()
    show_2d = st.checkbox("Mostrar PCA 2D (PC1 vs PC2)", value=True)
    opacity = st.slider("Opacidad puntos", 0.2, 1.0, 0.65)
    marker_size = st.slider("Tamaño puntos", 1, 8, 3)

# ============================================================
# 1) Load
# ============================================================
def load_df():
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv(default_path)

try:
    df = load_df().drop_duplicates()
except Exception as e:
    st.error("No se pudo cargar el CSV. Revisa la ruta (debe incluir `data/`) o sube el archivo.")
    st.exception(e)
    st.stop()

st.success(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]:,} columnas")

# Guardar IDs para hover
id_cols = [c for c in ["marca", "modelo"] if c in df.columns]
df_id = df[id_cols].copy() if id_cols else None

# ============================================================
# 2) Drop no estructurales (como notebook, sin tocar marca/modelo)
# ============================================================
cols_drop = ["impuestos_pagados", "precio_negociable", "recibe_vehiculo", "precio_usd"]
df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")

# ============================================================
# 3) Selección variables (igual notebook)
# ============================================================
numeric_candidates = [
    "precio_crc", "kilometraje", "antiguedad", "cilindrada", "puertas", "pasajeros", "participacion_mercado"
]
categorical_candidates = ["estilo", "combustible", "transmision", "segmento_marca", "origen_marca"]

numeric_features = [c for c in numeric_candidates if c in df.columns]
categorical_features = [c for c in categorical_candidates if c in df.columns]
selected_features = numeric_features + categorical_features

if not selected_features:
    st.error("No se encontraron variables candidatas en el CSV para PCA/Clustering.")
    st.stop()

df_model = df[selected_features].copy()

# Dropna críticos
critical_cols = [c for c in ["precio_crc", "kilometraje", "antiguedad"] if c in df_model.columns]
if critical_cols:
    before = df_model.shape[0]
    df_model = df_model.dropna(subset=critical_cols).copy()
    st.info(f"Dropna críticos {critical_cols}: {before:,} → {df_model.shape[0]:,}")

# ============================================================
# 4) Preprocess
# ============================================================
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

with st.spinner("Preprocesando (one-hot + escala)..."):
    X = preprocessor.fit_transform(df_model)
    X_dense = X.toarray() if hasattr(X, "toarray") else X

st.write(f"Matriz: **{X_dense.shape[0]:,} filas × {X_dense.shape[1]:,} features**")

# ============================================================
# 5) Clusters (usar existente o calcular)
# ============================================================
cluster_col = "cluster_id_hc"
labels = None

if cluster_col in df.columns and df[cluster_col].notna().any():
    # Alinear con df_model.index
    labels = df.loc[df_model.index, cluster_col].astype("Int64")
    st.success("Usando clusters existentes (`cluster_id_hc`) del CSV.")
else:
    if not compute_clusters:
        st.warning("El CSV no trae `cluster_id_hc` (o está vacío) y desactivaste el cálculo. Se mostrará PCA sin colores.")
    else:
        df_work = df_model.copy()
        X_work = X_dense

        if use_sampling and X_dense.shape[0] > max_rows:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_dense.shape[0], size=max_rows, replace=False)
            df_work = df_model.iloc[idx].copy()
            X_work = X_dense[idx]
            st.info(f"Ward calculado en muestra: {max_rows:,} filas (de {X_dense.shape[0]:,}).")

        with st.spinner("Calculando Ward (linkage)..."):
            # Nota: pdist+linkage(dist) como notebook. Para n grande es caro.
            dist = pdist(X_work, metric="euclidean")
            Z = linkage(dist, method="ward")
            labels_work = fcluster(Z, t=int(K), criterion="maxclust") - 1

        # Guardamos labels solo donde calculamos
        labels = pd.Series(pd.NA, index=df_model.index, dtype="Int64")
        labels.loc[df_work.index] = labels_work.astype(int)

        st.success(f"Clusters Ward generados (K={int(K)}).")
        st.dataframe(labels.dropna().astype(int).value_counts().sort_index().rename("count").to_frame(), use_container_width=True)

# ============================================================
# 6) PCA 3D
# ============================================================
with st.spinner("Entrenando PCA 3D..."):
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_dense)

expl = pca.explained_variance_ratio_
c1, c2, c3, c4 = st.columns(4)
c1.metric("PC1", f"{expl[0]*100:.2f}%")
c2.metric("PC2", f"{expl[1]*100:.2f}%")
c3.metric("PC3", f"{expl[2]*100:.2f}%")
c4.metric("Acumulada", f"{expl.sum()*100:.2f}%")

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"], index=df_model.index)

if labels is not None:
    df_pca[cluster_col] = labels

if df_id is not None:
    df_pca = df_pca.join(df_id, how="left")

hover_cols = [c for c in ["marca", "modelo"] if c in df_pca.columns]

st.subheader("PCA 3D")
fig3d = px.scatter_3d(
    df_pca,
    x="PC1", y="PC2", z="PC3",
    color=cluster_col if (cluster_col in df_pca.columns and df_pca[cluster_col].notna().any()) else None,
    opacity=opacity,
    hover_data=hover_cols,
    title=f"PCA 3D (color = {cluster_col if cluster_col in df_pca.columns else 'N/A'})"
)
fig3d.update_traces(marker=dict(size=marker_size))
st.plotly_chart(fig3d, use_container_width=True)

if show_2d:
    st.subheader("PCA 2D (PC1 vs PC2)")
    fig2d = px.scatter(
        df_pca,
        x="PC1", y="PC2",
        color=cluster_col if (cluster_col in df_pca.columns and df_pca[cluster_col].notna().any()) else None,
        opacity=opacity,
        hover_data=hover_cols,
        title="PC1 vs PC2"
    )
    st.plotly_chart(fig2d, use_container_width=True)

# Export opcional: PCA con labels
with st.container(border=True):
    st.subheader("⬇️ Export (opcional)")
    csv_bytes = df_pca.reset_index(drop=True).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar PCA (PC1,PC2,PC3 + cluster si existe)",
        data=csv_bytes,
        file_name="PCA_3D_with_clusters.csv",
        mime="text/csv"
    )
