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


# -------------------------------------------------------
# Carga de datos
# -------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_path = Path("data/CR_Autos.csv")

with st.container(border=True):
    st.title("📊 EDA | Comprensión de los datos")
    st.caption("Exploración interactiva del CV Normal (precio en colones CRC)")

if not data_path.exists():
    st.error("No se encontró data/CR_Autos.csv en el repositorio.")
    st.stop()

df = load_data(str(data_path))

# Asegurar tipos numéricos
numeric_cols = ["precio_crc", "kilometraje", "cilindrada", "pasajeros", "puertas", "antiguedad"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Derivar año desde antigüedad (solo para EDA)
if "antiguedad" in df.columns:
    current_year = datetime.now().year
    df["anio"] = current_year - df["antiguedad"]

st.write("")

# -------------------------------------------------------
# Sidebar filtros
# -------------------------------------------------------
st.sidebar.header("🎛️ Filtros")

def safe_filter(col):
    if col in df.columns:
        opts = sorted(df[col].dropna().astype(str).unique())
        return st.sidebar.multiselect(col.capitalize(), opts)
    return []

marca_f = safe_filter("marca")
prov_f = safe_filter("provincia")
comb_f = safe_filter("combustible")
trans_f = safe_filter("transmision")

if "anio" in df.columns and df["anio"].notna().any():
    min_y = int(df["anio"].min())
    max_y = int(df["anio"].max())
    year_range = st.sidebar.slider("Rango de año", min_y, max_y, (min_y, max_y))
else:
    year_range = None

fdf = df.copy()

if marca_f:
    fdf = fdf[fdf["marca"].astype(str).isin(marca_f)]
if prov_f:
    fdf = fdf[fdf["provincia"].astype(str).isin(prov_f)]
if comb_f:
    fdf = fdf[fdf["combustible"].astype(str).isin(comb_f)]
if trans_f:
    fdf = fdf[fdf["transmision"].astype(str).isin(trans_f)]
if year_range:
    fdf = fdf[(fdf["anio"] >= year_range[0]) & (fdf["anio"] <= year_range[1])]

# -------------------------------------------------------
# Resumen
# -------------------------------------------------------
with st.container(border=True):
    st.header("🧾 Resumen general")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(fdf):,}")
    if "precio_crc" in fdf.columns:
        c2.metric("Precio CRC (mediana)", f"{np.nanmedian(fdf['precio_crc']):,.0f}")
    if "kilometraje" in fdf.columns:
        c3.metric("Kilometraje (mediana)", f"{np.nanmedian(fdf['kilometraje']):,.0f}")

st.write("")

# -------------------------------------------------------
# Cantidad por año
# -------------------------------------------------------
with st.container(border=True):
    st.header("📅 Cantidad de vehículos por año")

    if "anio" in fdf.columns:
        counts = fdf.groupby("anio").size().reset_index(name="cantidad")

        if PLOTLY_OK:
            fig = px.bar(counts, x="anio", y="cantidad",
                         title="Cantidad de anuncios por año")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(counts.set_index("anio"))

        st.caption("Año derivado como: año_actual − antigüedad")
    else:
        st.info("No se pudo derivar el año.")

st.write("")

# -------------------------------------------------------
# Distribución de precios CRC
# -------------------------------------------------------
with st.container(border=True):
    st.header("💰 Distribución de precio (CRC)")

    if "precio_crc" in fdf.columns:
        tmp = fdf["precio_crc"].dropna()

        if PLOTLY_OK:
            fig = px.histogram(tmp, nbins=40,
                               title="Distribución de precios en CRC")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(tmp.value_counts(bins=40).sort_index())

st.write("")

# -------------------------------------------------------
# Precio vs Kilometraje
# -------------------------------------------------------
with st.container(border=True):
    st.header("🔁 Precio vs Kilometraje (CRC)")

    if all(c in fdf.columns for c in ["precio_crc", "kilometraje"]):
        sdf = fdf.dropna(subset=["precio_crc", "kilometraje"])

        if len(sdf) > 3000:
            sdf = sdf.sample(3000, random_state=42)

        if PLOTLY_OK:
            fig = px.scatter(
                sdf,
                x="kilometraje",
                y="precio_crc",
                color="provincia" if "provincia" in sdf.columns else None,
                title="Precio CRC vs Kilometraje"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(sdf[["kilometraje", "precio_crc"]])

st.caption("EDA realizado sobre el CV Normal utilizando precio en colones (CRC).")