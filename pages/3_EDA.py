import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import re
import unicodedata

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


@st.cache_data(show_spinner=False)
def load_geojson(uploaded_file) -> dict:
    return json.load(uploaded_file)


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


from typing import Optional


def cat_percentage_df(df: pd.DataFrame, col: str, top_n: Optional[int] = None) -> pd.DataFrame:
    s = df[col].dropna()
    pct = (s.value_counts(normalize=True) * 100)

    if top_n is not None and len(pct) > top_n:
        top = pct.head(top_n)
        others = 100 - top.sum()
        pct = pd.concat([top, pd.Series({"Otros": others})])

    out = pct.reset_index()
    out.columns = [col, "percentage"]
    return out


def normalize_text(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key_no_spaces(s: str) -> str:
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", "", s)
    return s


# Paleta mejorada (incluye interiores típicos)
COLOR_MAP = {
    # básicos
    "blanco": "#f2f2f2",
    "negro": "#111111",
    "gris": "#808080",
    "plata": "#c0c0c0",
    "plateado": "#c0c0c0",
    "azul": "#1f77b4",
    "rojo": "#d62728",
    "verde": "#2ca02c",
    "amarillo": "#ffdf00",
    "naranja": "#ff7f0e",
    "beige": "#d2b48c",
    "cafe": "#8b5a2b",
    "marron": "#8b5a2b",

    # interiores comunes / tonos
    "crema": "#ead9c0",
    "cuero": "#7a4a2a",
    "vino": "#6a0f2c",
    "terracota": "#c46a3c",

    # categorías operativas
    "otros": "#9e9e9e",
    "sin_dato": "#bdbdbd"
}


def to_base_color(cat):
    """
    Regla práctica: busca keywords dentro del texto y asigna una categoría base.
    Si no calza, cae en 'otros'. Si es NA, 'sin_dato'.
    """
    if pd.isna(cat):
        return "sin_dato"

    s = normalize_text(cat)
    if not s:
        return "sin_dato"

    rules = [
        ("blanco", ["blanc", "white", "perla", "ivory"]),
        ("negro", ["negr", "black", "ebony"]),
        ("gris", ["gris", "gray", "graphite", "grafito"]),
        ("plata", ["plata", "silver", "metallic"]),
        ("azul", ["azul", "blue", "navy", "marino"]),
        ("rojo", ["rojo", "red", "burgundy", "carmesi", "carmes", "vino"]),
        ("verde", ["verde", "green", "olive", "oliva"]),
        ("amarillo", ["amarill", "yellow", "gold"]),
        ("naranja", ["naranja", "orange"]),
        ("beige", ["beige", "tan", "camel", "arena"]),
        ("cafe", ["cafe", "marron", "brown", "chocolate", "cuero"]),
        ("crema", ["crema", "cream"]),
        ("terracota", ["terracota"]),
        ("vino", ["vino"]),
        ("cuero", ["cuero", "leather"]),
    ]

    for base, keys in rules:
        for k in keys:
            if k in s:
                return base

    return "otros"


# -------------------------------------------------------
# HERO
# -------------------------------------------------------
with st.container(border=True):
    st.title("📊 EDA | Comprensión de los datos")
    st.caption("Exploración interactiva del dataset. Sí, vamos a graficar todo lo que el notebook tenía y un poco más.")

st.write("")


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
with st.container(border=True):
    st.header("📁 Dataset")

    st.markdown("**Opción 1:** usar un archivo del repo. **Opción 2:** subir un CSV manualmente.")

    candidate_paths = [
        Path("data/CR_Autos.csv"),
        Path("data/CR_Autos_Cleaned_enriched.csv"),
        Path("CR_Autos_Cleaned_enriched.csv"),
        Path("CR_Autos.csv"),
    ]

    use_upload = st.toggle("Subir CSV manualmente", value=False)

    df = None
    if use_upload:
        up = st.file_uploader("Subí un CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.success("CSV cargado desde upload.")
        else:
            st.info("Subí un CSV para habilitar el EDA.")
    else:
        found = next((p for p in candidate_paths if p.exists()), None)
        if found is not None:
            df = load_data(str(found))
            st.success(f"CSV cargado desde el repo: {found}")
        else:
            st.warning("No encontré un CSV en rutas típicas. Activá el upload o subí el archivo al repo.")

    if df is None:
        st.stop()

df = coerce_numeric(df)
df = add_year_from_antiguedad(df)

st.write("")


# -------------------------------------------------------
# Sidebar filters + viz controls
# -------------------------------------------------------
st.sidebar.header("🎛️ Filtros")

# Preferir columnas normalizadas si existen
marca_col = "marca_norm" if "marca_norm" in df.columns else "marca"
marca_f = safe_multiselect(df, "Marca", marca_col)
prov_f  = safe_multiselect(df, "Provincia", "provincia")
comb_f  = safe_multiselect(df, "Combustible", "combustible")
trans_f = safe_multiselect(df, "Transmisión", "transmision")

estilo_f = safe_multiselect(df, "Estilo", "estilo") if "estilo" in df.columns else []

filters = {
    marca_col: marca_f,
    "provincia": prov_f,
    "combustible": comb_f,
    "transmision": trans_f,
    "estilo": estilo_f
}

fdf = apply_filters(df, filters)

# Year range (derived)
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
# Categorical % charts (from notebook)
# -------------------------------------------------------
with st.container(border=True):
    st.header("📌 Variables categóricas (frecuencia relativa %)")

    default_cols = [c for c in ["estilo", "combustible", "transmision", "estado"] if c in fdf.columns]
    if not default_cols:
        st.info("No encontré columnas categóricas típicas (estilo/combustible/transmision/estado) en este dataset.")
    else:
        cols = st.multiselect("Elegí columnas a graficar", options=default_cols, default=default_cols)
        top_n = st.slider("Top N categorías (para columnas con muchas categorías)", 5, 25, 10)
        for col in cols:
            df_pct = cat_percentage_df(fdf, col, top_n=top_n if col in ["estilo"] else None)
            df_pct = df_pct.sort_values("percentage", ascending=True)

            fig = px.bar(
                df_pct,
                x="percentage",
                y=col,
                orientation="h",
                text=df_pct["percentage"].round(1).astype(str) + "%",
                title=f"{col}: porcentaje por categoría",
                labels={"percentage": "Porcentaje (%)", col: col}
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_range=[0, max(5, float(df_pct["percentage"].max()) * 1.15)])
            st.plotly_chart(fig, use_container_width=True)

st.write("")


# -------------------------------------------------------
# Province distribution: bar + choropleth (if geojson)
# -------------------------------------------------------
with st.container(border=True):
    st.header("🗺️ Distribución por provincia")

    if "provincia" not in fdf.columns or fdf["provincia"].dropna().empty:
        st.info("No hay datos suficientes en 'provincia'.")
    else:
        prov_pct = (
            fdf["provincia"]
            .dropna()
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )
        prov_pct.columns = ["provincia", "percentage"]

        # Bar (siempre)
        pbar = prov_pct.sort_values("percentage", ascending=True)
        fig_bar = px.bar(
            pbar,
            x="percentage",
            y="provincia",
            orientation="h",
            text=pbar["percentage"].round(1).astype(str) + "%",
            title="Porcentaje de anuncios por provincia",
            labels={"percentage": "Porcentaje (%)", "provincia": "Provincia"},
        )
        fig_bar.update_traces(textposition="outside", cliponaxis=False)
        fig_bar.update_layout(xaxis_range=[0, max(5, float(pbar["percentage"].max()) * 1.15)])
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Mapa (opcional):** subí el GeoJSON de provincias (gadm u otro).")
        gj = st.file_uploader("GeoJSON de provincias (opcional)", type=["json", "geojson"])
        if gj is not None:
            try:
                cr_geo = load_geojson(gj)

                prov_pct["prov_key"] = prov_pct["provincia"].apply(norm_key_no_spaces)

                for feat in cr_geo.get("features", []):
                    props = feat.get("properties", {})
                    name = props.get("NAME_1") or props.get("name") or props.get("provincia") or ""
                    feat.setdefault("properties", {})["PROV_KEY"] = norm_key_no_spaces(name)

                fig_map = px.choropleth(
                    prov_pct,
                    geojson=cr_geo,
                    locations="prov_key",
                    featureidkey="properties.PROV_KEY",
                    color="percentage",
                    labels={"percentage": "Porcentaje (%)"},
                    title="Distribución porcentual de vehículos por provincia (Costa Rica)",
                )
                fig_map.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar el mapa con ese GeoJSON. Error: {e}")

st.write("")


# -------------------------------------------------------
# Market share pie (marca / marca_norm)
# -------------------------------------------------------
with st.container(border=True):
    st.header("🥧 Participación de mercado por marca")

    if marca_col not in fdf.columns or fdf[marca_col].dropna().empty:
        st.info(f"No hay datos suficientes en '{marca_col}'.")
    else:
        top_n = st.slider("Top N marcas (pie)", 5, 20, 8)
        share = fdf[marca_col].astype(str).value_counts(normalize=True).head(top_n)
        share["Otros"] = 1 - share.sum()

        market_df = share.reset_index()
        market_df.columns = ["marca", "participacion"]

        fig = px.pie(
            market_df,
            values="participacion",
            names="marca",
            title="Participación de mercado por marca",
            hole=0.0
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

st.write("")


# -------------------------------------------------------
# Price distribution: full, zoom, log (from notebook)
# -------------------------------------------------------
with st.container(border=True):
    st.header("💰 Distribución de precio (CRC)")

    if "precio_crc" not in fdf.columns or fdf["precio_crc"].dropna().empty:
        st.info("No hay datos suficientes en 'precio_crc'.")
    else:
        precio = fdf["precio_crc"].dropna()

        cA, cB, cC = st.columns(3)
        nbins = cA.slider("Bins", 10, 120, 50)
        zoom_max_m = cB.slider("Zoom (máx. millones CRC)", 10, 200, 60)
        show_log = cC.toggle("Mostrar histograma log10", value=True)

        # 1) Completo (visual cut opcional)
        cut = precio.quantile(pctl / 100.0)
        precio_cut = precio[precio <= cut]

        fig_full = px.histogram(
            x=precio_cut,
            nbins=nbins,
            title=f"Precio CRC (corte visual p{pctl})",
            labels={"x": "Precio (CRC)", "y": "Frecuencia"}
        )
        fig_full.update_layout(bargap=0.05, xaxis_title="Precio (CRC)", yaxis_title="Frecuencia")
        st.plotly_chart(fig_full, use_container_width=True)

        # 2) Zoom en millones
        precio_m = precio / 1_000_000
        precio_m = precio_m[precio_m <= zoom_max_m]

        fig_zoom = px.histogram(
            x=precio_m,
            nbins=max(10, int(nbins * 0.6)),
            title=f"Precio CRC (zoom hasta {zoom_max_m} millones)",
            labels={"x": "Precio (millones de CRC)", "y": "Frecuencia"}
        )
        fig_zoom.update_layout(bargap=0.05, xaxis=dict(range=[0, zoom_max_m]), yaxis_title="Frecuencia")
        st.plotly_chart(fig_zoom, use_container_width=True)

        # 3) Log10
        if show_log and (precio > 0).any():
            precio_pos = precio[precio > 0]
            precio_log10 = np.log10(precio_pos)

            fig_log = px.histogram(
                x=precio_log10,
                nbins=max(10, int(nbins * 0.6)),
                title="Distribución de precio_crc (escala log10)",
                labels={"x": "log10(precio_crc)", "y": "Frecuencia"}
            )
            fig_log.update_layout(bargap=0.05, xaxis_title="log10(precio_crc)", yaxis_title="Frecuencia")
            st.plotly_chart(fig_log, use_container_width=True)

st.write("")


# -------------------------------------------------------
# Cilindrada histogram (from notebook)
# -------------------------------------------------------
with st.container(border=True):
    st.header("🛠️ Distribución de cilindrada")

    if "cilindrada" not in fdf.columns or fdf["cilindrada"].dropna().empty:
        st.info("No hay datos suficientes en 'cilindrada'.")
    else:
        cil = fdf["cilindrada"].dropna()
        cut = cil.quantile(pctl / 100.0)
        cil = cil[cil <= cut]

        fig_cil = px.histogram(
            x=cil,
            nbins=40,
            title=f"Distribución de cilindrada (corte visual p{pctl})",
            labels={"x": "Cilindrada (cc)", "y": "Frecuencia"}
        )
        fig_cil.update_layout(bargap=0.05, xaxis_title="Cilindrada (cc)", yaxis_title="Frecuencia")
        st.plotly_chart(fig_cil, use_container_width=True)

st.write("")


# -------------------------------------------------------
# Colors exterior/interior (improved palette)
# -------------------------------------------------------
with st.container(border=True):
    st.header("🎨 Colores (exterior / interior)")

    col_ext = "color_exterior" if "color_exterior" in fdf.columns else None
    col_int = "color_interior" if "color_interior" in fdf.columns else None

    if not col_ext and not col_int:
        st.info("Este dataset no trae color_exterior / color_interior.")
    else:
        top_n = st.slider("Top N (colores)", 5, 25, 12)

        def plot_color(col_name: str, title: str):
            base = fdf[col_name].apply(to_base_color)
            pct = base.value_counts(normalize=True).mul(100).head(top_n)
            if pct.sum() < 99.9:
                pct["otros"] = 100 - pct.sum()

            cdf = pct.reset_index()
            cdf.columns = ["color_base", "percentage"]
            cdf["hex"] = cdf["color_base"].map(COLOR_MAP).fillna(COLOR_MAP["otros"])

            # px.bar con color discreto y orden bonito
            cdf = cdf.sort_values("percentage", ascending=True)

            fig = px.bar(
                cdf,
                x="percentage",
                y="color_base",
                orientation="h",
                text=cdf["percentage"].round(1).astype(str) + "%",
                title=title,
                labels={"percentage": "Porcentaje (%)", "color_base": "Color (base)"},
                color="color_base",
                color_discrete_map={k: COLOR_MAP.get(k, COLOR_MAP["otros"]) for k in cdf["color_base"].unique()}
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_range=[0, max(5, float(cdf["percentage"].max()) * 1.2)])
            st.plotly_chart(fig, use_container_width=True)

        if col_ext:
            plot_color(col_ext, "Color exterior (normalizado)")
        if col_int:
            plot_color(col_int, "Color interior (normalizado)")

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
# Price vs mileage
# -------------------------------------------------------
with st.container(border=True):
    st.header("🔁 Precio vs Kilometraje")

    if all(c in fdf.columns for c in ["precio_crc", "kilometraje"]) and fdf[["precio_crc", "kilometraje"]].dropna().shape[0] > 10:

        sdf = fdf.dropna(subset=["precio_crc", "kilometraje"]).copy()
        sdf["precio_millones"] = sdf["precio_crc"] / 1_000_000

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
            hover_data=[c for c in [marca_col, "modelo", "anio"] if c in sdf.columns],
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
# Top brands (count)
# -------------------------------------------------------
with st.container(border=True):
    st.header("🏷️ Top marcas (cantidad de anuncios)")

    if marca_col in fdf.columns and fdf[marca_col].notna().any():
        top_n = st.slider("Top N marcas (barras)", 5, 25, 12)

        counts = fdf[marca_col].astype(str).value_counts().head(top_n).reset_index()
        counts.columns = ["marca", "cantidad"]

        fig = px.bar(counts, x="marca", y="cantidad", title=f"Top {top_n} marcas")
        fig.update_layout(xaxis_title="Marca", yaxis_title="Cantidad de anuncios")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No hay datos suficientes en '{marca_col}'.")

st.caption("EDA interactivo listo para Streamlit. Si algo falta, casi seguro es porque el CSV no trae la columna, no porque yo tenga ganas de sufrir.")
