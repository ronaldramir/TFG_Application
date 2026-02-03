import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Preparación de datos | Ingeniería de variables",
    page_icon="🧩",
    layout="centered"
)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
with st.container(border=True):
    st.title("🧩 Preparación de datos e ingeniería de variables")
    st.caption("Transformación del CSV original a un dataset enriquecido para análisis y modelado")

st.write("")

# ------------------------------------------------------------
# Objetivo
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🎯 Objetivo")
    st.markdown(
        """
El objetivo de esta etapa es **mejorar la calidad y utilidad analítica** del dataset original,
creando variables derivadas que:
- reducen inconsistencias en categóricas (ej. marca, combustible, transmisión),
- incorporan **contexto de negocio** (origen y segmento de marca),
- y facilitan la interpretación de resultados (participación de mercado, top de marcas).

> Nota importante: en el notebook `Ingenieria_caracteristicas.ipynb` **no** se crea `premium_flag`.
"""
    )

st.write("")

# ------------------------------------------------------------
# Variables creadas
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧠 Variables nuevas (dataset enriched)")
    st.markdown(
        """
A partir del CSV original se generan las siguientes columnas:

- **`marca_norm`**: marca normalizada (mayúsculas + correcciones por alias).
- **`origen_marca`**: región de origen de la marca (mapeo por diccionario).
- **`segmento_marca`**: posicionamiento de mercado de la marca (mapeo por diccionario).
- **`participacion_mercado`**: porcentaje de participación de cada marca dentro del dataset.
- **`marca_topN`**: Top-N marcas más frecuentes; el resto se agrupa como **OTRAS**.
- **`combustible_norm`**: combustible estandarizado.
- **`transmision_norm`**: transmisión estandarizada.
"""
    )

st.write("")

# ------------------------------------------------------------
# Diccionarios (puedes sustituir por los tuyos exactos)
# ------------------------------------------------------------
# Correcciones/alias de marcas detectadas durante scraping/limpieza
brand_corrections = {
    "DONFENG": "DONGFENG",
    "SSANG": "SSANGYONG",
    "GREAT": "GREAT WALL",
    "DODGE/RAM": "DODGE",
    "LAND": "LAND ROVER",
    "ROVER": "LAND ROVER",
    "ROLLS": "ROLLS-ROYCE",
}

# Mapeos de negocio (ajusta según tus definiciones)
BRAND_ORIGIN = {
    "TOYOTA": "JAPON", "HONDA": "JAPON", "NISSAN": "JAPON", "MAZDA": "JAPON",
    "MITSUBISHI": "JAPON", "SUBARU": "JAPON", "SUZUKI": "JAPON",
    "LEXUS": "JAPON", "INFINITI": "JAPON",
    "KIA": "COREA", "HYUNDAI": "COREA", "SSANGYONG": "COREA",
    "FORD": "USA", "CHEVROLET": "USA", "JEEP": "USA", "DODGE": "USA",
    "BMW": "EUROPA", "MERCEDES-BENZ": "EUROPA", "AUDI": "EUROPA",
    "VOLKSWAGEN": "EUROPA", "VOLVO": "EUROPA", "LAND ROVER": "EUROPA",
    "PORSCHE": "EUROPA", "MINI": "EUROPA",
    "GEELY": "CHINA", "CHERY": "CHINA", "GREAT WALL": "CHINA", "DONGFENG": "CHINA",
}

BRAND_SEGMENT = {
    "BMW": "PREMIUM", "MERCEDES-BENZ": "PREMIUM", "AUDI": "PREMIUM",
    "LEXUS": "PREMIUM", "PORSCHE": "PREMIUM", "LAND ROVER": "PREMIUM",
    "VOLVO": "PREMIUM", "INFINITI": "PREMIUM",
    "TOYOTA": "MEDIO", "HONDA": "MEDIO", "NISSAN": "MEDIO", "MAZDA": "MEDIO",
    "SUBARU": "MEDIO", "VOLKSWAGEN": "MEDIO", "FORD": "MEDIO", "CHEVROLET": "MEDIO", "JEEP": "MEDIO",
    "KIA": "ECONOMICO", "HYUNDAI": "ECONOMICO", "SUZUKI": "ECONOMICO",
    "CHERY": "ECONOMICO", "GEELY": "ECONOMICO", "GREAT WALL": "ECONOMICO", "DONGFENG": "ECONOMICO",
}

FUEL_NORM = {
    "NAN": np.nan,
    "GASOLINA": "GASOLINA",
    "DIESEL": "DIESEL",
    "ELÉCTRICO": "ELECTRICO",
    "ELECTRICO": "ELECTRICO",
    "HIBRIDO": "HIBRIDO",
    "HÍBRIDO": "HIBRIDO",
}

TRANSMISSION_NORM = {
    "NAN": np.nan,
    "AUTOMATICA": "AUTOMATICA",
    "AUTOMÁTICA": "AUTOMATICA",
    "MANUAL": "MANUAL",
    "CVT": "CVT",
}

def _norm_text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna(pd.NA).str.strip().str.upper()

def feature_engineering(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Replica la lógica de ingeniería de características del notebook (robusta)."""
    df = df.copy()

    required = {"marca", "combustible", "transmision"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing)}")

    # marca_norm
    df["marca_norm"] = _norm_text(df["marca"]).replace("NAN", pd.NA)
    df["marca_norm"] = df["marca_norm"].replace(brand_corrections)

    # origen y segmento (fallback)
    df["origen_marca"] = df["marca_norm"].map(BRAND_ORIGIN).fillna("OTROS")
    df["segmento_marca"] = df["marca_norm"].map(BRAND_SEGMENT).fillna("MEDIO")

    # participación de mercado
    counts = df["marca_norm"].value_counts(dropna=False)
    total = len(df)
    df["participacion_mercado"] = df["marca_norm"].map((counts / total * 100).round(2))

    # marca_topN
    top_brands = set(df["marca_norm"].value_counts().head(top_n).index)
    df["marca_topN"] = df["marca_norm"].where(df["marca_norm"].isin(top_brands), "OTRAS")

    # combustible/transmisión normalizados
    df["combustible_norm"] = _norm_text(df["combustible"]).replace(FUEL_NORM)
    df["transmision_norm"] = _norm_text(df["transmision"]).replace(TRANSMISSION_NORM)

    return df

# ------------------------------------------------------------
# Lógica (con snippets)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧪 Lógica aplicada (fragmentos)")
    st.markdown("Los siguientes fragmentos resumen la transformación implementada:")

    st.code(
        """
# marca_norm + correcciones
df["marca_norm"] = df["marca"].astype(str).str.strip().str.upper().replace("NAN", np.nan)
df["marca_norm"] = df["marca_norm"].replace(brand_corrections)

# mapeos por diccionario
df["origen_marca"]   = df["marca_norm"].map(BRAND_ORIGIN).fillna("OTROS")
df["segmento_marca"] = df["marca_norm"].map(BRAND_SEGMENT).fillna("MEDIO")

# participación de mercado (%)
counts = df["marca_norm"].value_counts()
df["participacion_mercado"] = df["marca_norm"].map((counts/len(df)*100).round(2))

# Top-N marcas
top_brands = set(df["marca_norm"].value_counts().head(20).index)
df["marca_topN"] = df["marca_norm"].where(df["marca_norm"].isin(top_brands), "OTRAS")
        """.strip(),
        language="python"
    )

st.write("")

# ------------------------------------------------------------
# Demo interactivo (como en tu 2_Web_Scraping.py, pero con FE)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧰 Demo: generar dataset enriched")
    st.markdown("Carga el CSV original y genera el dataset enriquecido para descargarlo.")

    uploaded = st.file_uploader("Sube el CSV original (Unsupervised_Learning.csv)", type=["csv"])
    top_n = st.slider("Top N para `marca_topN`", 5, 50, 20, 1)

    st.write("")

    if uploaded is None:
        st.info("Sube un CSV para ejecutar la transformación.")
    else:
        df_raw = pd.read_csv(uploaded)

        try:
            df_enriched = feature_engineering(df_raw, top_n=top_n)
        except Exception as e:
            st.error(f"No se pudo aplicar feature engineering: {e}")
            st.stop()

        st.success(f"Listo: {df_enriched.shape[0]:,} filas | {df_enriched.shape[1]:,} columnas")

        st.markdown("**Vista previa (columnas clave):**")
        preview_cols = [c for c in [
            "marca", "modelo",
            "marca_norm", "origen_marca", "segmento_marca",
            "participacion_mercado", "marca_topN",
            "combustible_norm", "transmision_norm"
        ] if c in df_enriched.columns]
        st.dataframe(df_enriched[preview_cols].head(25), use_container_width=True)

        st.markdown("**Missing % en variables nuevas:**")
        new_cols = ["marca_norm","origen_marca","segmento_marca","participacion_mercado","marca_topN","combustible_norm","transmision_norm"]
        missing_pct = (df_enriched[new_cols].isna().mean()*100).sort_values(ascending=False).round(2)
        st.dataframe(missing_pct.to_frame("missing_%"), use_container_width=True)

        csv_bytes = df_enriched.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar Unsupervised_Learning_enriched.csv",
            data=csv_bytes,
            file_name="Unsupervised_Learning_enriched.csv",
            mime="text/csv",
        )
