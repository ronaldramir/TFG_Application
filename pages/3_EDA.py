import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import plotly.express as px


# -------------------------------------------------------
# Config
# -------------------------------------------------------
st.set_page_config(
    page_title="EDA | Comprensión de datos",
    page_icon="📊",
    layout="centered"
)


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["precio_crc", "kilometraje", "cilindrada", "pasajeros", "puertas", "antiguedad"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_year_from_antiguedad(df: pd.DataFrame) -> pd.DataFrame:
    if "antiguedad" in df.columns and df["antiguedad"].notna().any():
        current_year = datetime.now().year
        df["anio"] = current_year - df["antiguedad"]
    return df


def safe_multiselect(df: pd.DataFrame, label: str, col: str):
    if col in df.columns:
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
        return st.sidebar.multiselect(label, opts, default=[])
    return []


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    fdf = df.copy()
    for col, values in filters.items():
        if values and col in fdf.columns:
            fdf = fdf[fdf[col].astype(str).isin(values)]
    return fdf


def maybe_sample(df: pd.DataFrame, max_points=4000, seed=42):
    if len(df) > max_points:
        return df.sample(max_points, random_state=seed)
    return df


# -------------------------------------------------------
# HERO
# -------------------------------------------------------
with st.container(border=True):
    st.title("📊 EDA | Comprensión de los datos")
    st.caption("Exploración interactiva del CV Normal usando precios en colones (CRC)")
    st.markdown("Esta sección permite filtrar el dataset y generar gráficos para comprender el mercado antes del Feature Engineering.")

st.write("")


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
data_path = Path("data/CR_Autos.csv")

with st.container(border=True):
    st.header("📁 Dataset (CV Normal)")

    use_upload = st.toggle("Subir CSV manualmente (si el archivo no está en el repo)", value=False)

    df = None
    if use_upload:
        up = st.file_uploader("Subí CR_Autos.csv", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.success("CSV cargado desde upload.")
        else:
            st.info("Subí un CSV para habilitar el EDA.")
    else:
        if data_path.exists():
            df = load_data(str(data_path))
            st.success(f"CSV cargado desde el repo: {data_path}")
        else:
            st.warning("No encontré `data/CR_Autos.csv`. Activá el upload o subí el archivo al repo.")

    if df is None:
        st.stop()

df = coerce_numeric(df)
df = add_year_from_antiguedad(df)

st.write("")


# -------------------------------------------------------
# Sidebar filters + viz controls
# -------------------------------------------------------
st.sidebar.header("🎛️ Filtros")

marca_f = safe_multiselect(df, "Marca", "marca")
prov_f  = safe_multiselect(df, "Provincia", "provincia")
comb_f  = safe_multiselect(df, "Combustible", "combustible")
trans_f = safe_multiselect(df, "Transmisión", "transmision")

estilo_f = []
if "estilo" in df.columns:
    estilo_f = safe_multiselect(df, "Estilo", "estilo")

filters = {
    "marca": marca_f,
    "provincia": prov_f,
    "combustible": comb_f,
    "transmision": trans_f,
    "estilo": estilo_f
}

fdf = apply_filters(df, filters)

# Year range (derived)
year_range = None
if "anio" in fdf.columns and fdf["anio"].notna().any():
    min_y = int(np.nanmin(fdf["anio"]))
    max_y = int(np.nanmax(fdf["anio"]))
    year_range = st.sidebar.slider("Rango de año (derivado de antigüedad)", min_y, max_y, (min_y, max_y))
    fdf = fdf[(fdf["anio"] >= year_range[0]) & (fdf["anio"] <= year_range[1])]

st.sidebar.header("🧼 Visualización")
pctl = st.sidebar.slider("Corte de outliers (percentil para gráficos)", 90, 100, 99)
use_log = st.sidebar.toggle("Escala log (precio)", value=False)
show_data = st.sidebar.toggle("Mostrar tabla (primeras 50 filas)", value=False)

st.write("")


# -------------------------------------------------------
# Quick summary
# -------------------------------------------------------
with st.container(border=True):
    st.header("🧾 Resumen rápido")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Filas", f"{len(fdf):,}")
    c2.metric("Columnas", f"{fdf.shape[1]}")

    if "precio_crc" in fdf.columns and fdf["precio_crc"].notna().any():
        c3.metric("Precio CRC (mediana)", f"{np.nanmedian(fdf['precio_crc']):,.0f}")
    else:
        c3.metric("Precio CRC (mediana)", "N/A")

    if "kilometraje" in fdf.columns and fdf["kilometraje"].notna().any():
        c4.metric("Kilometraje (mediana)", f"{np.nanmedian(fdf['kilometraje']):,.0f}")
    else:
        c4.metric("Kilometraje (mediana)", "N/A")

    st.caption("Los filtros del sidebar afectan todos los gráficos.")

st.write("")


# -------------------------------------------------------
# Optional table + nulls
# -------------------------------------------------------
if show_data:
    with st.container(border=True):
        st.header("🔎 Vista del dataset")
        st.dataframe(fdf.head(50), use_container_width=True)

        st.subheader("Nulos por columna")
        nulls = fdf.isna().sum().sort_values(ascending=False)
        nulls_df = pd.DataFrame({"columna": nulls.index, "nulos": nulls.values})
        st.dataframe(nulls_df, use_container_width=True)

    st.write("")


# -------------------------------------------------------
# Vehicles by year
# -------------------------------------------------------
with st.container(border=True):
    st.header("📅 Cantidad de vehículos por año")

    if "anio" in fdf.columns and fdf["anio"].notna().any():
        counts = (
            fdf.dropna(subset=["anio"])
               .groupby("anio")
               .size()
               .reset_index(name="cantidad")
               .sort_values("anio")
        )

        fig = px.bar(counts, x="anio", y="cantidad", title="Cantidad de anuncios por año")
        fig.update_layout(xaxis_title="Año", yaxis_title="Cantidad de anuncios")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Año derivado como: año_actual − antigüedad (solo para EDA visual).")
    else:
        st.info("No se pudo derivar el año porque falta 'antiguedad' o está vacía.")

st.write("")


# -------------------------------------------------------
# Price distribution (correct histogram)
# -------------------------------------------------------
with st.container(border=True):
    st.header("💰 Distribución de precio (millones CRC)")

    if "precio_crc" in fdf.columns and fdf["precio_crc"].notna().any():

        precio_millones = (fdf["precio_crc"] / 1_000_000).dropna()

        # Outliers only for visualization
        cut = precio_millones.quantile(pctl / 100.0)
        precio_millones = precio_millones[precio_millones <= cut]

        if use_log:
            precio_plot = np.log1p(precio_millones)
            xlabel = "log(1 + precio en millones)"
        else:
            precio_plot = precio_millones
            xlabel = "Precio (millones CRC)"

        fig = px.histogram(
            x=precio_plot,
            nbins=40,
            title="Distribución de precio"
        )
        fig.update_layout(xaxis_title=xlabel, yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No hay datos suficientes en 'precio_crc'.")

st.write("")


# -------------------------------------------------------
# Price vs mileage (separate section, clean)
# -------------------------------------------------------
with st.container(border=True):
    st.header("🔁 Precio vs Kilometraje")

    if all(c in fdf.columns for c in ["precio_crc", "kilometraje"]) and fdf[["precio_crc", "kilometraje"]].dropna().shape[0] > 10:

        sdf = fdf.dropna(subset=["precio_crc", "kilometraje"]).copy()
        sdf["precio_millones"] = sdf["precio_crc"] / 1_000_000

        # Outliers only for visualization
        cut = sdf["precio_millones"].quantile(pctl / 100.0)
        sdf = sdf[sdf["precio_millones"] <= cut]

        if use_log:
            sdf["precio_plot"] = np.log1p(sdf["precio_millones"])
            ylabel = "log(1 + precio en millones)"
        else:
            sdf["precio_plot"] = sdf["precio_millones"]
            ylabel = "Precio (millones CRC)"

        sdf = maybe_sample(sdf, max_points=4000)

        fig = px.scatter(
            sdf,
            x="kilometraje",
            y="precio_plot",
            color="provincia" if "provincia" in sdf.columns else None,
            hover_data=[c for c in ["marca", "modelo", "anio"] if c in sdf.columns],
            opacity=0.55,
            title="Relación Precio vs Kilometraje"
        )
        fig.update_layout(xaxis_title="Kilometraje", yaxis_title=ylabel)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No hay suficientes datos para graficar precio_crc vs kilometraje.")

st.write("")


# -------------------------------------------------------
# Price by year (boxplot)
# -------------------------------------------------------
with st.container(border=True):
    st.header("📦 Precio por año (boxplot)")

    if "anio" in fdf.columns and "precio_crc" in fdf.columns and fdf[["anio", "precio_crc"]].dropna().shape[0] > 10:
        bdf = fdf.dropna(subset=["anio", "precio_crc"]).copy()
        bdf["precio_millones"] = bdf["precio_crc"] / 1_000_000

        cut = bdf["precio_millones"].quantile(pctl / 100.0)
        bdf = bdf[bdf["precio_millones"] <= cut]

        fig = px.box(bdf, x="anio", y="precio_millones", title="Distribución de precio por año")
        fig.update_layout(xaxis_title="Año", yaxis_title="Precio (millones CRC)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay suficientes datos para boxplot por año.")

st.write("")


# -------------------------------------------------------
# Top brands
# -------------------------------------------------------
with st.container(border=True):
    st.header("🏷️ Top marcas (cantidad de anuncios)")

    if "marca" in fdf.columns and fdf["marca"].notna().any():
        top_n = st.slider("Top N marcas", 5, 25, 12)

        counts = fdf["marca"].astype(str).value_counts().head(top_n).reset_index()
        counts.columns = ["marca", "cantidad"]

        fig = px.bar(counts, x="marca", y="cantidad", title=f"Top {top_n} marcas")
        fig.update_layout(xaxis_title="Marca", yaxis_title="Cantidad de anuncios")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos suficientes en 'marca'.")

st.caption("EDA interactivo sobre CV Normal usando precio en colones (CRC). El Feature Engineering se presenta en la siguiente sección.")