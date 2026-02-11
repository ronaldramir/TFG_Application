import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Perfilado de Clusters", page_icon="📊", layout="wide")

with st.container(border=True):
    st.title("📊 Perfilado de Clusters: Radar + Barras")
    st.caption("Se asume que el dataset ya trae una columna de cluster (por defecto: cluster_id_hc).")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Controles")

    uploaded = st.file_uploader("Cargar CSV con clusters (opcional)", type=["csv"])
    default_path = st.text_input(
        "Ruta CSV por defecto",
        value="data/Unsupervised_Learning_HC_WARD_K3.csv"
    )

    CLUSTER_COL = st.text_input("Columna de cluster", value="cluster_id_hc")
    TOP_K_CATS = st.slider("Top K categorías (barras)", 3, 15, 8)

    agg = st.selectbox("Agregación numérica", ["mean", "median"], index=1)

    st.divider()
    show_table = st.checkbox("Mostrar tabla resumen numérica", value=True)

# ============================================================
# Load
# ============================================================
def load_df():
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv(default_path)

try:
    df = load_df()
except Exception as e:
    st.error("No se pudo cargar el CSV. Revisa la ruta (debe incluir `data/`) o sube el archivo.")
    st.exception(e)
    st.stop()

if CLUSTER_COL not in df.columns:
    st.error(f"No existe la columna de clusters: '{CLUSTER_COL}'.")
    st.stop()

df = df.dropna(subset=[CLUSTER_COL]).copy()
df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)

st.success(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]:,} columnas | clusters: {sorted(df[CLUSTER_COL].unique())}")

# ============================================================
# Variables candidatas (igual notebook)
# ============================================================
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

num_vars = [c for c in NUM_VARS_CANDIDATES if c in df.columns]
cat_vars = [c for c in CAT_VARS_CANDIDATES if c in df.columns]

# ============================================================
# 1) Radar (numéricas)
# ============================================================
with st.container(border=True):
    st.header("🕸️ Radar (variables numéricas)")
    if not num_vars:
        st.warning("No hay variables numéricas candidatas en el CSV.")
    else:
        picked = st.multiselect("Variables numéricas", num_vars, default=num_vars)
        if picked:
            # Tabla resumen
            grp = df.groupby(CLUSTER_COL)[picked]
            summary = grp.mean() if agg == "mean" else grp.median()

            # Normalizar 0..1 para radar (porque escalas diferentes = radar inútil)
            norm = summary.copy()
            for col in picked:
                mn, mx = float(norm[col].min()), float(norm[col].max())
                if mx - mn == 0:
                    norm[col] = 0.0
                else:
                    norm[col] = (norm[col] - mn) / (mx - mn)

            # Plotly radar
            fig = go.Figure()
            cats = picked + [picked[0]]  # cerrar el polígono

            for cl in norm.index:
                vals = norm.loc[cl, picked].tolist()
                vals = vals + [vals[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=cats,
                    fill="toself",
                    name=f"Cluster {cl}"
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=520,
                title=f"Radar normalizado (0–1) usando {agg}"
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_table:
                st.subheader(f"Tabla {agg} (sin normalizar)")
                st.dataframe(summary, use_container_width=True)

        else:
            st.info("Selecciona al menos una variable numérica.")

# ============================================================
# 2) Barras (categóricas)
# ============================================================
with st.container(border=True):
    st.header("📦 Barras por cluster (variables categóricas)")
    if not cat_vars:
        st.warning("No hay variables categóricas candidatas en el CSV.")
    else:
        cat_pick = st.selectbox("Variable categórica", cat_vars, index=0)

        # Top categorías globales para mantener comparabilidad entre clusters
        top_cats = df[cat_pick].astype(str).value_counts().head(TOP_K_CATS).index.tolist()
        dfx = df.copy()
        dfx[cat_pick] = dfx[cat_pick].astype(str).where(dfx[cat_pick].astype(str).isin(top_cats), other="OTROS")

        # Conteos y % por cluster
        ct = (
            dfx.groupby([CLUSTER_COL, cat_pick])
               .size()
               .reset_index(name="count")
        )
        totals = ct.groupby(CLUSTER_COL)["count"].transform("sum")
        ct["pct"] = ct["count"] / totals * 100.0

        # Barras apiladas en %
        fig = px.bar(
            ct,
            x=CLUSTER_COL,
            y="pct",
            color=cat_pick,
            title=f"Distribución % por cluster: {cat_pick} (Top {TOP_K_CATS} + OTROS)",
            labels={CLUSTER_COL: "Cluster", "pct": "%"}
        )
        fig.update_layout(barmode="stack", height=520)
        st.plotly_chart(fig, use_container_width=True)

        # Barras por cluster: top dentro de cada cluster (vista alternativa)
        st.subheader("Top categorías por cluster (tabla)")
        top_by_cluster = (
            ct.sort_values([CLUSTER_COL, "pct"], ascending=[True, False])
              .groupby(CLUSTER_COL)
              .head(TOP_K_CATS)
        )
        st.dataframe(top_by_cluster, use_container_width=True)

# ============================================================
# 3) Export opcional: tabla perfilado
# ============================================================
with st.container(border=True):
    st.header("⬇️ Export (opcional)")

    if num_vars:
        grp = df.groupby(CLUSTER_COL)[num_vars]
        summary = grp.mean() if agg == "mean" else grp.median()
        csv_bytes = summary.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Descargar tabla numérica ({agg})",
            data=csv_bytes,
            file_name=f"cluster_profile_numeric_{agg}.csv",
            mime="text/csv"
        )
    else:
        st.info("No hay variables numéricas para exportar.")
