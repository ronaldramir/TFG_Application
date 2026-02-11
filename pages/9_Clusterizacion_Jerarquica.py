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
# Configuración de página
# ============================================================
st.set_page_config(
    page_title="Clustering Jerárquico (Ward)",
    page_icon="🌳",
    layout="wide"
)

with st.container(border=True):
    st.title("🌳 Clusterización Jerárquica (Ward) + Dendrograma")
    st.caption("Versión Streamlit del notebook: flujo PCA-like → preprocesamiento → Ward → dendrograma → corte por K → export CSV")

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

@st.cache_data(show_spinner=False)
def preprocess_matrix(df_model: pd.DataFrame, numeric_features, categorical_features):
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X = preprocessor.fit_transform(df_model)
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    return X_dense

@st.cache_data(show_spinner=False)
def compute_linkage(X_dense: np.ndarray):
    # Nota: linkage es O(n^2). Si n es grande, se va a poner dramático.
    return linkage(X_dense, method="ward")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Controles")

    uploaded = st.file_uploader("Cargar CSV (opcional)", type=["csv"])
    default_path = st.text_input(
        "Ruta CSV por defecto (si no subes archivo)",
        value="data/CR_Autos_Cleaned_enriched.csv"
    )

    st.divider()

    # Dendrograma
    st.subheader("Dendrograma")
    trunc_p = st.slider("p (truncate_mode='lastp')", 10, 120, 40)
    font_size = st.slider("Tamaño fuente hojas", 6, 14, 10)
    show_contracted = st.toggle("show_contracted", value=True)

    st.divider()

    # K
    st.subheader("Corte por K")
    K = st.slider("K (maxclust)", 2, 10, 3)

    st.divider()

    # Rendimiento
    st.subheader("Rendimiento")
    use_sampling = st.toggle(
        "Usar muestreo para Ward",
        value=False,
        help="Ward + linkage escala mal. Si tu dataset es grande, esto evita que tu compu llore."
    )
    max_rows = st.slider("Máximo filas (si muestreo)", 500, 15000, 6000, step=500)

    st.divider()
    show_sample = st.checkbox("Mostrar muestra del dataset", value=True)
    show_selected = st.checkbox("Mostrar variables seleccionadas", value=True)

# ============================================================
# 1) Carga
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
# 2) Limpieza mínima (igual notebook)
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
# 3) Selección de variables (idéntico notebook)
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

categorical_candidates = [
    "estilo",
    "combustible",
    "transmision",
    "segmento_marca",
    "origen_marca",
]

numeric_features = [c for c in numeric_candidates if c in df.columns]
categorical_features = [c for c in categorical_candidates if c in df.columns]
selected_features = numeric_features + categorical_features

with st.container(border=True):
    st.header("🧩 Variables seleccionadas para Ward")

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
# 4) Dropna críticos (igual notebook)
# ============================================================
with st.container(border=True):
    st.header("🩹 Faltantes críticos")

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
# 5) Preprocesamiento + Ward
# ============================================================
with st.container(border=True):
    st.header("🌳 Ward + Dendrograma")

    # Muestreo opcional para que linkage no se vuelva una ópera trágica
    df_work = df_model.copy()
    if use_sampling and df_work.shape[0] > max_rows:
        df_work = df_work.sample(n=max_rows, random_state=42)
        st.info(f"Usando muestreo: {max_rows:,} filas (de {df_model.shape[0]:,}).")

    with st.spinner("Preprocesando (One-Hot + escala)..."):
        X_dense = preprocess_matrix(df_work, numeric_features, categorical_features)

    st.write(f"✅ Matriz lista: **{X_dense.shape[0]:,} filas × {X_dense.shape[1]:,} features** (después de One-Hot)")

    # Linkage
    with st.spinner("Calculando linkage Ward (esto es lo pesado)..."):
        Z = compute_linkage(X_dense)

    st.write(f"✅ Linkage generado: **{Z.shape[0]:,}** uniones")

    # Dendrograma
    fig = plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=trunc_p,
        leaf_rotation=90,
        leaf_font_size=font_size,
        show_contracted=show_contracted
    )
    plt.title(f"Dendrograma (Ward) — truncado (p={trunc_p})")
    plt.xlabel("Clusters fusionados (truncado)")
    plt.ylabel("Distancia (Ward)")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.write("")

# ============================================================
# 6) Corte por K + export
# ============================================================
with st.container(border=True):
    st.header("✂️ Corte del dendrograma y export")

    labels_hc = fcluster(Z, t=int(K), criterion="maxclust") - 1  # 0..K-1
    counts = pd.Series(labels_hc).value_counts().sort_index()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Distribución de clusters")
        st.dataframe(counts.rename("count").to_frame(), use_container_width=True)

    with c2:
        st.subheader("Notas")
        st.markdown("""
- El corte usa **criterion='maxclust'**, igual que el notebook.
- Si estás en modo **muestreo**, estos labels corresponden solo a las filas muestreadas.
- Si quieres **guardar clusters para TODO el dataset**, desactiva muestreo (o sube max_rows).
""")

    # Construir df de salida
    out = df_work.copy()
    out["cluster_id_hc"] = labels_hc

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV con cluster_id_hc",
        data=csv_bytes,
        file_name=f"Unsupervised_Learning_HC_WARD_K{int(K)}.csv",
        mime="text/csv"
    )

st.write("")
with st.container(border=True):
    st.header("🧾 Resumen rápido")
    st.markdown(f"""
- Features numéricas: **{len(numeric_features)}**
- Features categóricas: **{len(categorical_features)}**
- Filas usadas: **{df_work.shape[0]:,}**
- K seleccionado: **{int(K)}**
""")
