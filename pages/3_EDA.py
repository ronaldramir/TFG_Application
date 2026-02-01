import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="EDA | Comprensión de datos", page_icon="📊", layout="centered")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que numéricas sean numéricas (sin inventar nada, solo casteo seguro)."""
    num_cols = ["precio_crc", "precio_usd", "cilindrada", "pasajeros", "kilometraje", "puertas", "antiguedad"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_year_from_antiguedad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva 'anio' desde 'antiguedad' para gráficos por año.
    Criterio: anio = año_actual - antiguedad
    (EDA: solo para visualización; FE formal va después).
    """
    if "antiguedad" in df.columns:
        current_year = datetime.now().year
        df["anio"] = current_year - df["antiguedad"]
    return df


# ----------------------------
# HERO
# ----------------------------
with st.container(border=True):
    st.title("📊 EDA | Comprensión de los datos")
    st.caption("Exploración interactiva del CV Normal (antes del Feature Engineering)")

st.write("")

# ----------------------------
# Carga de datos: repo o upload
# ----------------------------
with st.container(border=True):
    st.header("📁 Carga del dataset (CV Normal)")

    default_path = Path("data/CR_Autos.csv")
    use_upload = st.toggle("Subir archivo CSV manualmente (si no está en el repo)", value=False)

    df = None

    if use_upload:
        up = st.file_uploader("Subí el CV Normal (.csv)", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.success("CSV cargado desde upload.")
        else:
            st.info("Subí un CSV para habilitar el EDA.")
    else:
        if default_path.exists():
            df = load_csv(str(default_path))
            st.success(f"CSV cargado desde el repo: {default_path}")
        else:
            st.warning("No encontré `data/Unsupervised_Learning.csv` en el repo. Activá el upload o subí el archivo al repo.")

    if df is None:
        st.stop()

df = coerce_numeric(df)
df = add_year_from_antiguedad(df)

st.write("")

# ----------------------------
# Sidebar filtros
# ----------------------------
st.sidebar.header("🎛️ Filtros")

# Filtros seguros (si existen columnas)
def safe_multiselect(label, col):
    if col in df.columns:
        opts = sorted([x for x in df[col].dropna().astype(str).unique().tolist()])
        return st.sidebar.multiselect(label, opts, default=[])
    return []

selected_prov = safe_multiselect("Provincia", "provincia")
selected_marca = safe_multiselect("Marca", "marca")
selected_comb = safe_multiselect("Combustible", "combustible")
selected_trans = safe_multiselect("Transmisión", "transmision")

# Rango de años si existe 'anio'
if "anio" in df.columns and df["anio"].notna().any():
    anio_min = int(df["anio"].dropna().min())
    anio_max = int(df["anio"].dropna().max())
    year_range = st.sidebar.slider("Rango de año (derivado de antigüedad)", anio_min, anio_max, (anio_min, anio_max))
else:
    year_range = None

# Aplicar filtros
fdf = df.copy()

if selected_prov:
    fdf = fdf[fdf["provincia"].astype(str).isin(selected_prov)]
if selected_marca:
    fdf = fdf[fdf["marca"].astype(str).isin(selected_marca)]
if selected_comb:
    fdf = fdf[fdf["combustible"].astype(str).isin(selected_comb)]
if selected_trans:
    fdf = fdf[fdf["transmision"].astype(str).isin(selected_trans)]
if year_range and "anio" in fdf.columns:
    fdf = fdf[(fdf["anio"] >= year_range[0]) & (fdf["anio"] <= year_range[1])]

# ----------------------------
# Resumen rápido
# ----------------------------
with st.container(border=True):
    st.header("🧾 Resumen rápido")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(fdf):,}")
    c2.metric("Columnas", f"{fdf.shape[1]}")
    if "precio_usd" in fdf.columns:
        c3.metric("Precio USD (mediana)", f"{np.nanmedian(fdf['precio_usd']):,.0f}")
    if "kilometraje" in fdf.columns:
        c4.metric("Km (mediana)", f"{np.nanmedian(fdf['kilometraje']):,.0f}")

st.write("")

# ----------------------------
# Tabla + nulos
# ----------------------------
with st.container(border=True):
    st.header("🔎 Vista del dataset")
    st.dataframe(fdf.head(50), use_container_width=True)

    st.subheader("Nulos por columna")
    nulls = fdf.isna().sum().sort_values(ascending=False)
    nulls_df = pd.DataFrame({"columna": nulls.index, "nulos": nulls.values})
    st.dataframe(nulls_df, use_container_width=True)

st.write("")

# ----------------------------
# Gráfico: cantidad de carros por año
# ----------------------------
with st.container(border=True):
    st.header("📅 Cantidad de vehículos por año")

    if "anio" not in fdf.columns or not fdf["anio"].notna().any():
        st.info("No pude derivar 'anio' porque falta 'antiguedad' o está vacía.")
    else:
        counts = (
            fdf.dropna(subset=["anio"])
               .groupby("anio")
               .size()
               .reset_index(name="cantidad")
               .sort_values("anio")
        )

        if PLOTLY_OK:
            fig = px.bar(counts, x="anio", y="cantidad", title="Cantidad de anuncios por año (derivado)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(counts.set_index("anio")["cantidad"])

        st.caption("Año derivado como: año_actual − antigüedad (solo para EDA visual).")

st.write("")

# ----------------------------
# Distribuciones: precio y km
# ----------------------------
with st.container(border=True):
    st.header("📈 Distribuciones")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Precio (USD)")
        if "precio_usd" in fdf.columns and fdf["precio_usd"].notna().any():
            tmp = fdf["precio_usd"].dropna()
            if PLOTLY_OK:
                fig = px.histogram(tmp, nbins=40, title="Distribución de precio USD")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(tmp.value_counts(bins=40).sort_index())
        else:
            st.info("No hay datos suficientes en precio_usd.")

    with colB:
        st.subheader("Kilometraje")
        if "kilometraje" in fdf.columns and fdf["kilometraje"].notna().any():
            tmp = fdf["kilometraje"].dropna()
            if PLOTLY_OK:
                fig = px.histogram(tmp, nbins=40, title="Distribución de kilometraje")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(tmp.value_counts(bins=40).sort_index())
        else:
            st.info("No hay datos suficientes en kilometraje.")

st.write("")

# ----------------------------
# Scatter: precio vs km
# ----------------------------
with st.container(border=True):
    st.header("🔁 Relación: Precio vs Kilometraje")

    if all(c in fdf.columns for c in ["precio_usd", "kilometraje"]) and fdf[["precio_usd", "kilometraje"]].dropna().shape[0] > 10:
        sdf = fdf.dropna(subset=["precio_usd", "kilometraje"]).copy()

        # Submuestreo opcional para performance
        max_points = 3000
        if len(sdf) > max_points:
            sdf = sdf.sample(max_points, random_state=42)

        color_col = "provincia" if "provincia" in sdf.columns else None

        if PLOTLY_OK:
            fig = px.scatter(
                sdf,
                x="kilometraje",
                y="precio_usd",
                color=color_col,
                hover_data=[c for c in ["marca", "modelo", "anio"] if c in sdf.columns],
                title="Precio USD vs Kilometraje"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(sdf[["kilometraje", "precio_usd"]])
    else:
        st.info("No hay suficientes datos numéricos para graficar precio_usd vs kilometraje.")

st.caption("EDA interactivo sobre CV Normal. El Feature Engineering se presenta en la siguiente sección.")