import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Perfilado de Clusters (Radar + Barras)", page_icon="📊", layout="wide")

DEFAULT_PATH = "data/Unsupervised_Learning_HC_WARD_K3.csv"
DEFAULT_CLUSTER_COL = "cluster_id_hc"

NUM_VARS_CANDIDATES = [
    "precio_crc",
    "kilometraje",
    "antiguedad",
    "cilindrada",
    "puertas",
    "pasajeros",
    "participacion_mercado"
]

CAT_VARS_CANDIDATES = [
    "estilo",
    "combustible",
    "transmision",
    "segmento_marca",
    "origen_marca"
]

# Colores fijos (misma idea que en PCA)
COLOR_MAP = {
    "Cluster 0": "#636EFA",
    "Cluster 1": "#EF553B",
    "Cluster 2": "#00CC96",
    "Cluster 3": "#AB63FA",
    "Cluster 4": "#FFA15A",
    "Cluster 5": "#19D3F3",
}

# ============================================================
# UI
# ============================================================
with st.container(border=True):
    st.title("📊 Perfilado de Clusters")
    st.caption("Radar (z-score) + Barras numéricas (log1p / 0–1) + Barras categóricas (% dentro del cluster).")

with st.sidebar:
    st.header("⚙️ Controles")

    uploaded = st.file_uploader("Cargar CSV con clusters (opcional)", type=["csv"])
    path = st.text_input("Ruta CSV por defecto", value=DEFAULT_PATH)

    st.divider()
    CLUSTER_COL = st.text_input("Columna de cluster", value=DEFAULT_CLUSTER_COL)
    TOP_K_CATS = st.slider("Top K categorías (barras)", 3, 20, 8)

    st.divider()
    st.subheader("Numéricas")
    agg = st.selectbox("Agregación", ["mean", "median"], index=0)  # promedio por defecto
    log1p_cols = st.multiselect(
        "Aplicar log1p a (si existen)",
        options=["precio_crc", "kilometraje", "cilindrada"],
        default=["precio_crc", "kilometraje", "cilindrada"]
    )

    st.divider()
    show_tables = st.checkbox("Mostrar tablas resumen", value=True)
    show_raw = st.checkbox("Mostrar muestra del dataset", value=False)

# ============================================================
# Load
# ============================================================
def load_df():
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv(path)

try:
    df = load_df()
except Exception as e:
    st.error("No se pudo cargar el CSV. Revisa la ruta o sube el archivo.")
    st.exception(e)
    st.stop()

if CLUSTER_COL not in df.columns:
    st.error(f"No existe la columna '{CLUSTER_COL}' en el CSV.")
    st.stop()

df = df[df[CLUSTER_COL].notna()].copy()
df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)
df["cluster_label"] = df[CLUSTER_COL].apply(lambda x: f"Cluster {x}")

st.success(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]:,} columnas | clusters: {sorted(df[CLUSTER_COL].unique())}")

if show_raw:
    st.dataframe(df.head(30), use_container_width=True)

num_vars = [c for c in NUM_VARS_CANDIDATES if c in df.columns]
cat_vars = [c for c in CAT_VARS_CANDIDATES if c in df.columns]

# ============================================================
# 1) Perfil numérico (mean/median)
# ============================================================
with st.container(border=True):
    st.header("1) Perfil numérico por cluster (agregado)")

    if not num_vars:
        st.warning("No hay variables numéricas candidatas en el CSV.")
    else:
        picked = st.multiselect("Variables numéricas", num_vars, default=num_vars)

        if picked:
            grp = df.groupby(CLUSTER_COL)[picked]
            profile = grp.mean() if agg == "mean" else grp.median()

            if show_tables:
                st.subheader(f"Tabla ({agg})")
                st.dataframe(profile, use_container_width=True)

        else:
            profile = pd.DataFrame()
            st.info("Selecciona al menos una variable numérica.")

# ============================================================
# 2) Radar Z-score (recomendado)
# ============================================================
with st.container(border=True):
    st.header("2) Radar por cluster (z-score de promedios numéricos)")

    if num_vars and not profile.empty:
        z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
        z = z.fillna(0)

        fig = go.Figure()
        theta = list(z.columns)

        for cid in z.index:
            label = f"Cluster {int(cid)}"
            fig.add_trace(go.Scatterpolar(
                r=z.loc[cid].values,
                theta=theta,
                fill="toself",
                name=label,
                line=dict(color=COLOR_MAP.get(label, None))
            ))

        fig.update_layout(
            height=520,
            title=f"Radar (z-score) usando {agg}",
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay variables numéricas para radar (o no seleccionaste variables).")

# ============================================================
# 3) Barras numéricas (log1p recomendado)
# ============================================================
with st.container(border=True):
    st.header("3) Barras numéricas por cluster (log1p en variables sesgadas)")

    if num_vars and not profile.empty:
        df_num = df[[CLUSTER_COL] + list(profile.columns)].copy()

        for col in log1p_cols:
            if col in df_num.columns:
                # log1p requiere >= -1; si hay basura negativa, la arreglamos de forma segura
                df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
                df_num[col] = df_num[col].clip(lower=0)
                df_num[col] = np.log1p(df_num[col])

        grp = df_num.groupby(CLUSTER_COL)[list(profile.columns)]
        profile_log = grp.mean() if agg == "mean" else grp.median()
        long_log = profile_log.reset_index().melt(id_vars=[CLUSTER_COL], var_name="variable", value_name="valor")

        # Para colores consistentes, coloreamos por label
        long_log["cluster_label"] = long_log[CLUSTER_COL].apply(lambda x: f"Cluster {int(x)}")

        fig = px.bar(
            long_log,
            x=CLUSTER_COL,
            y="valor",
            color="cluster_label",
            facet_col="variable",
            facet_col_wrap=3,
            title=f"{agg} por cluster (log1p aplicado a: {', '.join([c for c in log1p_cols if c in df_num.columns]) or 'ninguna'})",
            color_discrete_map=COLOR_MAP
        )
        fig.update_layout(showlegend=True, height=560)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay variables numéricas para barras (o no seleccionaste variables).")

# ============================================================
# 4) Barras numéricas (normalizadas 0–1)
# ============================================================
with st.container(border=True):
    st.header("4) Barras numéricas por cluster (normalizadas 0–1)")

    if num_vars and not profile.empty:
        denom = (profile.max() - profile.min()).replace(0, np.nan)
        prof_norm = ((profile - profile.min()) / denom).fillna(0)
        long_norm = prof_norm.reset_index().melt(id_vars=[CLUSTER_COL], var_name="variable", value_name="valor")
        long_norm["cluster_label"] = long_norm[CLUSTER_COL].apply(lambda x: f"Cluster {int(x)}")

        fig = px.bar(
            long_norm,
            x=CLUSTER_COL,
            y="valor",
            color="cluster_label",
            facet_col="variable",
            facet_col_wrap=3,
            title=f"{agg} por cluster (normalizado 0–1)",
            color_discrete_map=COLOR_MAP
        )
        fig.update_layout(showlegend=True, height=560)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay variables numéricas para barras normalizadas (o no seleccionaste variables).")

# ============================================================
# 5) Barras categóricas (% dentro del cluster)
# ============================================================
def plot_categorical_distribution(df_in: pd.DataFrame, cluster_col: str, cat_col: str, top_k: int):
    tmp = df_in[[cluster_col, cat_col]].dropna().copy()
    tmp[cat_col] = tmp[cat_col].astype(str)
    tmp["cluster_label"] = tmp[cluster_col].apply(lambda x: f"Cluster {int(x)}")

    # Top categorías globales
    top_cats = tmp[cat_col].value_counts().head(top_k).index.tolist()
    tmp = tmp[tmp[cat_col].isin(top_cats)]

    ct = pd.crosstab(tmp["cluster_label"], tmp[cat_col], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars=["cluster_label"], var_name=cat_col, value_name="porcentaje")

    fig = px.bar(
        ct,
        x="cluster_label",
        y="porcentaje",
        color=cat_col,
        barmode="stack",
        title=f"Distribución (%) de {cat_col} por cluster (Top {top_k})",
    )
    fig.update_layout(yaxis_title="% dentro del cluster", height=520)
    return fig, ct

with st.container(border=True):
    st.header("5) Categóricas por cluster (% dentro del cluster)")

    if not cat_vars:
        st.warning("No hay variables categóricas candidatas en el CSV.")
    else:
        cat_pick = st.selectbox("Variable categórica", cat_vars, index=0)
        fig, ct_tbl = plot_categorical_distribution(df, CLUSTER_COL, cat_pick, TOP_K_CATS)
        st.plotly_chart(fig, use_container_width=True)

        if show_tables:
            st.subheader("Tabla (porcentaje)")
            st.dataframe(ct_tbl.sort_values(["cluster_label", "porcentaje"], ascending=[True, False]), use_container_width=True)

# ============================================================
# Export opcional
# ============================================================
with st.container(border=True):
    st.header("⬇️ Export (opcional)")

    if num_vars and not profile.empty:
        csv_bytes = profile.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Descargar tabla numérica ({agg})",
            data=csv_bytes,
            file_name=f"cluster_profile_numeric_{agg}.csv",
            mime="text/csv"
        )
    else:
        st.info("No hay tabla numérica disponible para exportar (selecciona variables numéricas primero).")
