import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PERFILADO (Streamlit) replicando EXACTAMENTE tu flujo:
# 1) Cargar enriched
# 2) Dropear cols no estructurales (sin tocar marca/modelo)
# 3) Selección variables (num + cat)
# 4) dropna críticos
# 5) Preprocesamiento (median + scaler + one-hot)
# 6) Ward (pdist euclidean + linkage ward + fcluster maxclust) -> cluster_id_hc
# 7) Perfilado: profile_mean -> Radar z-score + barras
# ============================================================

st.set_page_config(page_title="Ward + Perfilado (idéntico al notebook)", page_icon="📊", layout="wide")

# Colores fijos para consistencia
COLOR_MAP = {
    "Cluster 0": "#636EFA",
    "Cluster 1": "#EF553B",
    "Cluster 2": "#00CC96",
    "Cluster 3": "#AB63FA",
    "Cluster 4": "#FFA15A",
    "Cluster 5": "#19D3F3",
}

NUM_VARS_CANDIDATES = [
    "precio_crc", "kilometraje", "antiguedad", "cilindrada", "puertas", "pasajeros", "participacion_mercado"
]
CAT_VARS_CANDIDATES = ["estilo", "combustible", "transmision", "segmento_marca", "origen_marca"]

with st.container(border=True):
    st.title("📊 Ward + Perfilado (replica notebook)")
    st.caption("Esto recalcula Ward sobre el mismo espacio preprocesado y luego genera radar/barras con los mismos pasos del notebook.")

with st.sidebar:
    st.header("⚙️ Controles")

    uploaded = st.file_uploader("Cargar CSV (opcional)", type=["csv"])
    path = st.text_input("Ruta CSV por defecto", value="data/CR_Autos_Cleaned_enriched.csv")

    st.divider()
    K = st.slider("K (Ward)", 2, 8, 3)

    TOP_K_CATS = st.slider("Top K categorías (barras categóricas)", 3, 20, 8)

    st.divider()
    st.subheader("Radar / agregación")
    agg = st.selectbox("Agregación numérica", ["mean", "median"], index=0)  # mean por defecto (como tu bloque)
    radar_scale = st.radio("Escala radar", ["z-score", "0–1 (min-max)"], index=0)

    st.divider()
    st.subheader("Barras numéricas")
    log1p_cols = st.multiselect(
        "Aplicar log1p a (si existen)",
        options=["precio_crc", "kilometraje", "cilindrada"],
        default=["precio_crc", "kilometraje", "cilindrada"]
    )

    st.divider()
    show_tables = st.checkbox("Mostrar tablas resumen", value=True)
    show_sample = st.checkbox("Mostrar muestra del dataset", value=False)

# ------------------------------------------------------------
# 1) Cargar dataset
# ------------------------------------------------------------
try:
    df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(path)
    df = df.drop_duplicates()
except Exception as e:
    st.error("No se pudo cargar el CSV. Revisa la ruta (debe existir `data/...`) o sube el archivo.")
    st.exception(e)
    st.stop()

st.success(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]:,} columnas")

# ------------------------------------------------------------
# 2) Drop columnas no estructurales (NO marca/modelo)
# ------------------------------------------------------------
cols_drop = ["impuestos_pagados", "precio_negociable", "recibe_vehiculo", "precio_usd"]
df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")

# Guardar IDs (hover)
id_cols = [c for c in ["marca", "modelo"] if c in df.columns]
df_id = df[id_cols].copy() if id_cols else None

if show_sample:
    st.dataframe(df.head(30), use_container_width=True)

# ------------------------------------------------------------
# 3) Selección de variables
# ------------------------------------------------------------
numeric_features = [c for c in NUM_VARS_CANDIDATES if c in df.columns]
categorical_features = [c for c in CAT_VARS_CANDIDATES if c in df.columns]
selected_features = numeric_features + categorical_features

if not selected_features:
    st.error("No se encontraron variables candidatas (num/cat) en el CSV.")
    st.stop()

df_model = df[selected_features].copy()

# ------------------------------------------------------------
# 4) Dropna críticos
# ------------------------------------------------------------
critical_cols = [c for c in ["precio_crc", "kilometraje", "antiguedad"] if c in df_model.columns]
if critical_cols:
    before = df_model.shape[0]
    df_model = df_model.dropna(subset=critical_cols).copy()
    st.info(f"Dropna críticos {critical_cols}: {before:,} → {df_model.shape[0]:,}")

# ------------------------------------------------------------
# 5) Preprocesamiento (median + scaler + one-hot)
# ------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

with st.spinner("Preprocesando (one-hot + escala)..."):
    X = preprocessor.fit_transform(df_model)
    X_dense = X.toarray() if hasattr(X, "toarray") else X

st.write(f"X_dense: **{X_dense.shape[0]:,} filas × {X_dense.shape[1]:,} features**")

# ------------------------------------------------------------
# 6) Ward sobre el MISMO espacio (pdist + linkage + fcluster)
# ------------------------------------------------------------
with st.spinner("Calculando Ward (esto puede tardar con ~11k filas)..."):
    dist = pdist(X_dense, metric="euclidean")
    Z = linkage(dist, method="ward")
    labels_hc = fcluster(Z, t=int(K), criterion="maxclust") - 1  # 0..K-1

st.success("Ward listo.")
st.dataframe(pd.Series(labels_hc).value_counts().sort_index().rename("count").to_frame(), use_container_width=True)

# Asignar clusters alineado por índice original
df_plot = df.loc[df_model.index].copy()
df_plot["cluster_id_hc"] = labels_hc.astype(int)
df_plot["cluster_label"] = df_plot["cluster_id_hc"].apply(lambda x: f"Cluster {x}")

# ------------------------------------------------------------
# 7) Perfil numérico (mean como tu bloque)
# ------------------------------------------------------------
num_vars = [c for c in NUM_VARS_CANDIDATES if c in df_plot.columns]
cat_vars = [c for c in CAT_VARS_CANDIDATES if c in df_plot.columns]

with st.container(border=True):
    st.header("1) Perfil numérico por cluster")

    if not num_vars:
        st.warning("No hay variables numéricas para perfilado.")
        profile = pd.DataFrame()
    else:
        picked = st.multiselect("Variables numéricas", num_vars, default=num_vars)
        if picked:
            grp = df_plot.groupby("cluster_id_hc")[picked]
            profile = grp.mean() if agg == "mean" else grp.median()

            if show_tables:
                st.subheader(f"Tabla ({agg})")
                st.dataframe(profile, use_container_width=True)
        else:
            profile = pd.DataFrame()
            st.info("Selecciona al menos una variable numérica.")

# ------------------------------------------------------------
# 8) Radar (z-score) igual que tu bloque
# ------------------------------------------------------------
with st.container(border=True):
    st.header("2) Radar por cluster")

    if not profile.empty:
        if radar_scale == "z-score":
            radar = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
            radar = radar.fillna(0)
            title = f"Radar por cluster (z-score de {agg})"
            m = float(np.nanmax(np.abs(radar.values))) if radar.size else 1.0
            r_range = [-max(1.0, m), max(1.0, m)]
        else:
            denom = (profile.max() - profile.min()).replace(0, np.nan)
            radar = ((profile - profile.min()) / denom).fillna(0)
            title = f"Radar por cluster (0–1 min-max de {agg})"
            r_range = [0, 1]

        fig = go.Figure()
        for cid in radar.index:
            label = f"Cluster {int(cid)}"
            fig.add_trace(go.Scatterpolar(
                r=radar.loc[cid].values,
                theta=radar.columns,
                fill="toself",
                name=label,
                line=dict(color=COLOR_MAP.get(label))
            ))

        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True, range=r_range)),
            showlegend=True,
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay profile numérico para radar.")

# ------------------------------------------------------------
# 9) Barras numéricas (VALORES ABSOLUTOS)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("3B) Barras numéricas (valores absolutos)")

    if not profile.empty:

        profile_abs = profile.copy().reset_index()
        profile_abs["cluster_label"] = profile_abs["cluster_id_hc"].apply(lambda x: f"Cluster {int(x)}")

        long_abs = profile_abs.melt(
            id_vars=["cluster_id_hc", "cluster_label"],
            var_name="variable",
            value_name="valor_real"
        )

        fig_abs = px.bar(
            long_abs,
            x="cluster_id_hc",
            y="valor_real",
            color="cluster_label",
            facet_col="variable",
            facet_col_wrap=3,
            title=f"{agg} por cluster (valores reales)",
            color_discrete_map=COLOR_MAP
        )

        fig_abs.update_layout(showlegend=True, height=560)
        st.plotly_chart(fig_abs, use_container_width=True)

        if show_tables:
            st.subheader("Tabla valores reales")
            st.dataframe(profile, use_container_width=True)

    else:
        st.info("No hay profile numérico para barras absolutas.")



# ------------------------------------------------------------
# 10) Barras numéricas (log1p) igual que tu bloque
# ------------------------------------------------------------
with st.container(border=True):
    st.header("3) Barras numéricas (log1p)")

    if not profile.empty:
        df_num = df_plot[["cluster_id_hc"] + list(profile.columns)].copy()

        # EXACTO como tu bloque: log1p directo (sin clip), asumiendo valores >= 0
        for col in log1p_cols:
            if col in df_num.columns:
                df_num[col] = np.log1p(pd.to_numeric(df_num[col], errors="coerce"))

        grp = df_num.groupby("cluster_id_hc")[list(profile.columns)]
        profile_log = grp.mean() if agg == "mean" else grp.median()

        long_log = profile_log.reset_index().melt(
            id_vars=["cluster_id_hc"],
            var_name="variable",
            value_name="valor_log1p"
        )
        long_log["cluster_label"] = long_log["cluster_id_hc"].apply(lambda x: f"Cluster {int(x)}")

        fig = px.bar(
            long_log,
            x="cluster_id_hc",
            y="valor_log1p",
            color="cluster_label",
            facet_col="variable",
            facet_col_wrap=3,
            title="Promedios numéricos por cluster (log1p en variables seleccionadas)",
            color_discrete_map=COLOR_MAP
        )
        fig.update_layout(showlegend=True, height=560)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay profile numérico para barras log1p.")

# ------------------------------------------------------------
# 11) Barras numéricas (0–1) igual que tu bloque
# ------------------------------------------------------------
with st.container(border=True):
    st.header("4) Barras numéricas (normalizadas 0–1)")

    if not profile.empty:
        denom = (profile.max() - profile.min()).replace(0, np.nan)
        profile_norm = ((profile - profile.min()) / denom).fillna(0)

        long_norm = profile_norm.reset_index().melt(
            id_vars=["cluster_id_hc"],
            var_name="variable",
            value_name="valor_0_1"
        )
        long_norm["cluster_label"] = long_norm["cluster_id_hc"].apply(lambda x: f"Cluster {int(x)}")

        fig = px.bar(
            long_norm,
            x="cluster_id_hc",
            y="valor_0_1",
            color="cluster_label",
            facet_col="variable",
            facet_col_wrap=3,
            title="Promedios numéricos por cluster (normalizado 0–1)",
            color_discrete_map=COLOR_MAP
        )
        fig.update_layout(showlegend=True, height=560)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay profile numérico para barras 0–1.")

# ------------------------------------------------------------
# 12) Barras categóricas (%) igual que tu bloque
# ------------------------------------------------------------
with st.container(border=True):
    st.header("5) BARRAS CATEGÓRICAS (% dentro del cluster)")

    if not cat_vars:
        st.warning("No hay variables categóricas para barras.")
    else:
        cat_pick = st.selectbox("Variable categórica", cat_vars, index=0)

        tmp = df_plot[["cluster_id_hc", cat_pick]].dropna().copy()
        tmp[cat_pick] = tmp[cat_pick].astype(str)

        top_cats = tmp[cat_pick].value_counts().head(int(TOP_K_CATS)).index.tolist()
        tmp = tmp[tmp[cat_pick].isin(top_cats)]

        ct = pd.crosstab(tmp["cluster_id_hc"], tmp[cat_pick], normalize="index") * 100
        ct = ct.reset_index().melt(id_vars=["cluster_id_hc"], var_name=cat_pick, value_name="porcentaje")

        fig = px.bar(
            ct,
            x="cluster_id_hc",
            y="porcentaje",
            color=cat_pick,
            barmode="stack",
            title=f"Distribución (%) de {cat_pick} por cluster (Top {TOP_K_CATS})"
        )
        fig.update_layout(yaxis_title="% dentro del cluster", height=520)
        st.plotly_chart(fig, use_container_width=True)

        if show_tables:
            st.subheader("Tabla (%)")
            st.dataframe(ct.sort_values(["cluster_id_hc", "porcentaje"], ascending=[True, False]), use_container_width=True)

# ------------------------------------------------------------
# Export opcional
# ------------------------------------------------------------
with st.container(border=True):
    st.header("⬇️ Export (opcional)")
    out = df_plot.copy()
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar dataset con cluster_id_hc (solo filas usadas en Ward)",
        data=csv_bytes,
        file_name=f"Unsupervised_Learning_HC_WARD_K{int(K)}_from_enriched.csv",
        mime="text/csv"
    )
