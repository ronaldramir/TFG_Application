import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ============================================================
# Configuración
# ============================================================
st.set_page_config(
    page_title="Ward + Dendrograma",
    layout="wide"
)

st.title("Clusterización Jerárquica (Ward)")

st.markdown("""
Esta sección asume que:
- El dataset ya fue limpiado previamente.
- El CSV se encuentra en el subfolder `data/`.
- Solo nos interesa ejecutar Ward, visualizar el dendrograma y cortar por K.
""")

# ============================================================
# 1) Carga directa desde /data
# ============================================================

DATA_PATH = "data/CR_Autos_Cleaned_enriched.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"Dataset cargado desde {DATA_PATH} → {df.shape[0]:,} filas × {df.shape[1]:,} columnas")
except Exception as e:
    st.error("No se pudo cargar el archivo desde /data.")
    st.exception(e)
    st.stop()

# ============================================================
# 2) Selección de variables (idéntica al notebook)
# ============================================================

numeric_features = [
    "precio_crc",
    "kilometraje",
    "antiguedad",
    "cilindrada",
    "puertas",
    "pasajeros",
    "participacion_mercado",
]

categorical_features = [
    "estilo",
    "combustible",
    "transmision",
    "segmento_marca",
    "origen_marca",
]

numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

selected_features = numeric_features + categorical_features

if len(selected_features) == 0:
    st.error("No se encontraron variables válidas para Ward.")
    st.stop()

df_model = df[selected_features].copy()

# Drop críticos (igual que notebook)
critical_cols = [c for c in ["precio_crc", "kilometraje", "antiguedad"] if c in df_model.columns]
if len(critical_cols) > 0:
    df_model = df_model.dropna(subset=critical_cols)

st.write(f"Filas usadas para clustering: {df_model.shape[0]:,}")

# ============================================================
# 3) Preprocesamiento
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

with st.spinner("Preprocesando datos..."):
    X = preprocessor.fit_transform(df_model)
    X_dense = X.toarray() if hasattr(X, "toarray") else X

st.write(f"Matriz final: {X_dense.shape[0]:,} filas × {X_dense.shape[1]:,} features")

# ============================================================
# 4) Ward + Dendrograma
# ============================================================

st.subheader("Dendrograma (Ward)")

with st.spinner("Calculando linkage Ward..."):
    Z = linkage(X_dense, method="ward")

fig = plt.figure(figsize=(14, 6))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=40,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)
plt.title("Dendrograma (Ward) - Vista truncada")
plt.xlabel("Clusters fusionados")
plt.ylabel("Distancia (Ward)")
plt.tight_layout()

st.pyplot(fig, clear_figure=True)

# ============================================================
# 5) Corte por K
# ============================================================

st.subheader("Corte por número de clusters")

K = st.slider("Seleccionar K", 2, 10, 3)

labels_hc = fcluster(Z, t=int(K), criterion="maxclust") - 1

cluster_counts = pd.Series(labels_hc).value_counts().sort_index()

st.write("Distribución de clusters:")
st.dataframe(cluster_counts.rename("count").to_frame(), use_container_width=True)

# ============================================================
# 6) Export
# ============================================================

df_out = df_model.copy()
df_out["cluster_id_hc"] = labels_hc

csv_bytes = df_out.to_csv(index=False).encode("utf-8")

st.download_button(
    "Descargar CSV con cluster_id_hc",
    data=csv_bytes,
    file_name=f"Unsupervised_Learning_HC_WARD_K{int(K)}.csv",
    mime="text/csv"
)
