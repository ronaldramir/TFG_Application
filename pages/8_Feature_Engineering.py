# 4_Feature_Engineering.py
# ------------------------------------------------------------
# Streamlit Page: Preparación de Datos e Ingeniería de Variables
# (Basado en Ingenieria_caracteristicas.ipynb)
#
# Qué hace:
# - Carga el CSV original (o permite subirlo)
# - Aplica la ingeniería de características del notebook
# - Muestra el resultado y permite descargar el CSV enriquecido
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st

PAGE_TITLE = "Preparación de Datos e Ingeniería de Variables"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("Implementación en Streamlit basada en el notebook: Ingenieria_caracteristicas.ipynb")

# =========================
# Diccionarios del notebook
# =========================

# Correcciones puntuales de marca detectadas en el dataset
brand_corrections = {'MERCEDES-BENZ': 'MERCEDES', 'MERCEDES BENZ': 'MERCEDES', 'BENZ': 'MERCEDES', 'ALFA ROMEO': 'ALFA', 'RANGE ROVER': 'LAND ROVER', 'DISCOVERY': 'LAND ROVER', 'GREATWALL': 'GREAT WALL'}

# Mapeo marca -> origen (del notebook)
BRAND_ORIGIN = {'TOYOTA': 'JAPON', 'HONDA': 'JAPON', 'NISSAN': 'JAPON', 'MAZDA': 'JAPON', 'MITSUBISHI': 'JAPON', 'SUBARU': 'JAPON', 'SUZUKI': 'JAPON', 'DAIHATSU': 'JAPON', 'ISUZU': 'JAPON', 'HINO': 'JAPON', 'LEXUS': 'JAPON', 'INFINITI': 'JAPON', 'ACURA': 'JAPON', 'SCION': 'JAPON', 'DATSUN': 'JAPON', 'HYUNDAI': 'COREA', 'KIA': 'COREA', 'GENESIS': 'COREA', 'DAEWOO': 'COREA', 'SAMSUNG': 'COREA', 'SSANGYONG': 'COREA', 'FORD': 'USA', 'CHEVROLET': 'USA', 'GMC': 'USA', 'CADILLAC': 'USA', 'BUICK': 'USA', 'CHRYSLER': 'USA', 'JEEP': 'USA', 'DODGE': 'USA', 'PONTIAC': 'USA', 'OLDSMOBILE': 'USA', 'HUMMER': 'USA', 'GEO': 'USA', 'TESLA': 'USA', 'FREIGHTLINER': 'USA', 'INTERNATIONAL': 'USA', 'MACK': 'USA', 'PETERBILT': 'USA', 'BMW': 'EUROPA', 'MERCEDES': 'EUROPA', 'VOLKSWAGEN': 'EUROPA', 'AUDI': 'EUROPA', 'PORSCHE': 'EUROPA', 'MINI': 'EUROPA', 'SKODA': 'EUROPA', 'VOLVO': 'EUROPA', 'PEUGEOT': 'EUROPA', 'RENAULT': 'EUROPA', 'CITROEN': 'EUROPA', 'FIAT': 'EUROPA', 'ALFA': 'EUROPA', 'FERRARI': 'EUROPA', 'LAMBORGHINI': 'EUROPA', 'MASERATI': 'EUROPA', 'BENTLEY': 'EUROPA', 'ROLLS-ROYCE': 'EUROPA', 'JAGUAR': 'EUROPA', 'LAND ROVER': 'EUROPA', 'ROVER': 'EUROPA', 'PIAGGIO': 'EUROPA', 'MG': 'EUROPA', 'BYD': 'CHINA', 'CHERY': 'CHINA', 'CHANGAN': 'CHINA', 'BAIC': 'CHINA', 'GEELY': 'CHINA', 'JAC': 'CHINA', 'FAW': 'CHINA', 'DONGFENG': 'CHINA', 'HAVAL': 'CHINA', 'GREAT WALL': 'CHINA', 'FOTON': 'CHINA', 'JETOUR': 'CHINA', 'OMODA': 'CHINA', 'KAIYI': 'CHINA', 'SOUEAST': 'CHINA', 'VGV': 'CHINA', 'AION': 'CHINA', 'NETA': 'CHINA', 'ZEEKR': 'CHINA', 'XPENG': 'CHINA', 'XIAOMI': 'CHINA', 'MAXUS': 'CHINA', 'HIGER': 'CHINA', 'SHINERAY': 'CHINA', 'BAW': 'CHINA', 'MAHINDRA': 'INDIA', 'CMC': 'ASIA_OTROS', 'BLUEBIRD': 'OTROS', 'ZAP': 'OTROS'}

# Mapeo marca -> segmento (del notebook)
BRAND_SEGMENT = {'AUDI': 'PREMIUM', 'BMW': 'PREMIUM', 'MERCEDES': 'PREMIUM', 'LEXUS': 'PREMIUM', 'INFINITI': 'PREMIUM', 'GENESIS': 'PREMIUM', 'JAGUAR': 'PREMIUM', 'LAND ROVER': 'PREMIUM', 'PORSCHE': 'PREMIUM', 'VOLVO': 'PREMIUM', 'MINI': 'PREMIUM', 'ALFA': 'PREMIUM', 'BENTLEY': 'PREMIUM', 'FERRARI': 'PREMIUM', 'LAMBORGHINI': 'PREMIUM', 'MASERATI': 'PREMIUM', 'ROLLS-ROYCE': 'PREMIUM', 'CADILLAC': 'PREMIUM', 'TESLA': 'PREMIUM', 'ZEEKR': 'PREMIUM', 'XPENG': 'PREMIUM', 'XIAOMI': 'PREMIUM', 'CHERY': 'ECONOMICO', 'CHANGAN': 'ECONOMICO', 'BAIC': 'ECONOMICO', 'JAC': 'ECONOMICO', 'GEELY': 'ECONOMICO', 'DONGFENG': 'ECONOMICO', 'FAW': 'ECONOMICO', 'SOUEAST': 'ECONOMICO', 'VGV': 'ECONOMICO', 'JETOUR': 'ECONOMICO', 'OMODA': 'ECONOMICO', 'KAIYI': 'ECONOMICO', 'NETA': 'ECONOMICO', 'AION': 'ECONOMICO', 'SHINERAY': 'ECONOMICO', 'BAW': 'ECONOMICO', 'HINO': 'COMERCIAL', 'ISUZU': 'COMERCIAL', 'FOTON': 'COMERCIAL', 'FREIGHTLINER': 'COMERCIAL', 'INTERNATIONAL': 'COMERCIAL', 'MACK': 'COMERCIAL', 'PETERBILT': 'COMERCIAL', 'HIGER': 'COMERCIAL', 'MAXUS': 'COMERCIAL', 'JMC': 'COMERCIAL', 'TOYOTA': 'MEDIO', 'HONDA': 'MEDIO', 'NISSAN': 'MEDIO', 'MAZDA': 'MEDIO', 'HYUNDAI': 'MEDIO', 'KIA': 'MEDIO', 'MITSUBISHI': 'MEDIO', 'SUBARU': 'MEDIO', 'SUZUKI': 'MEDIO', 'VOLKSWAGEN': 'MEDIO', 'FORD': 'MEDIO', 'CHEVROLET': 'MEDIO', 'GMC': 'MEDIO', 'JEEP': 'MEDIO', 'CHRYSLER': 'MEDIO', 'DODGE': 'MEDIO', 'RENAULT': 'MEDIO', 'PEUGEOT': 'MEDIO', 'CITROEN': 'MEDIO', 'FIAT': 'MEDIO', 'BYD': 'MEDIO', 'MG': 'MEDIO', 'GREAT WALL': 'MEDIO', 'HAVAL': 'MEDIO', 'DAIHATSU': 'MEDIO', 'DATSUN': 'MEDIO', 'SCION': 'MEDIO', 'DAEWOO': 'MEDIO', 'SAMSUNG': 'MEDIO', 'MAHINDRA': 'MEDIO', 'BUICK': 'MEDIO', 'PONTIAC': 'MEDIO', 'OLDSMOBILE': 'MEDIO', 'HUMMER': 'MEDIO', 'GEO': 'MEDIO', 'PIAGGIO': 'MEDIO'}

# Normalización de combustible (del notebook)
FUEL_NORM = {
    "NAN": np.nan,
    "GASOLINA": "GASOLINA",
    "DIESEL": "DIESEL",
    "ELÉCTRICO": "ELECTRICO",
    "ELECTRICO": "ELECTRICO",
    "HIBRIDO": "HIBRIDO",
    "HÍBRIDO": "HIBRIDO",
}

# Normalización de transmisión (del notebook)
TRANSMISSION_NORM = {
    "NAN": np.nan,
    "AUTOMATICA": "AUTOMATICA",
    "AUTOMÁTICA": "AUTOMATICA",
    "MANUAL": "MANUAL",
    "CVT": "CVT",
}

# =========================
# Función principal (replica notebook)
# =========================
def feature_engineering(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    df = df.copy()

    required_cols = {"marca", "combustible", "transmision"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing)}")

    # 1) Normalización de marca
    df["marca_norm"] = (
        df["marca"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("NAN", np.nan)
    )
    df["marca_norm"] = df["marca_norm"].replace(brand_corrections)

    # 2) Origen de marca
    df["origen_marca"] = df["marca_norm"].map(BRAND_ORIGIN).fillna("OTROS")

    # 3) Segmento de marca
    df["segmento_marca"] = df["marca_norm"].map(BRAND_SEGMENT).fillna("MEDIO")

    # 4) Participación de mercado (%) por marca_norm
    marca_counts = df["marca_norm"].value_counts()
    total_registros = len(df)
    marca_participacion = (marca_counts / total_registros * 100).round(2)
    df["participacion_mercado"] = df["marca_norm"].map(marca_participacion)

    # 5) Agrupación Top-N marcas
    top_brands = set(df["marca_norm"].value_counts().head(top_n).index)
    df["marca_topN"] = df["marca_norm"].where(df["marca_norm"].isin(top_brands), "OTRAS")

    # 6) Normalización de combustible
    df["combustible_norm"] = (
        df["combustible"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(FUEL_NORM)
    )

    # 7) Normalización de transmisión
    df["transmision_norm"] = (
        df["transmision"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(TRANSMISSION_NORM)
    )

    return df


# =========================
# UI: Entrada de datos
# =========================
st.subheader("Entrada de datos")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Opción A:** subir el CSV original")
    uploaded = st.file_uploader("Sube Unsupervised_Learning.csv", type=["csv"])

with col2:
    st.markdown("**Opción B:** usar una ruta por defecto (si existe en tu repo)")
    default_path = st.text_input("Ruta del CSV original", value="Unsupervised_Learning.csv")

top_n = st.slider("Top N marcas para `marca_topN`", min_value=5, max_value=50, value=20, step=1)

# Cargar df
df_raw = None
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    try:
        df_raw = pd.read_csv(default_path)
    except Exception:
        df_raw = None

if df_raw is None:
    st.warning("No se pudo cargar el CSV. Sube el archivo o ajusta la ruta.")
    st.stop()

st.success("CSV cargado: {:,} filas | {:,} columnas".format(df_raw.shape[0], df_raw.shape[1]))

# =========================
# Explicación (adaptada del notebook)
# =========================
st.subheader("Descripción del proceso")

st.markdown(
    """
Este módulo replica la lógica de **Ingenieria_caracteristicas.ipynb**, enriqueciendo el dataset original con variables derivadas que
mejoran la calidad y la interpretabilidad del análisis.

**Variables creadas:**
- `marca_norm`: marca normalizada (mayúsculas + correcciones).
- `origen_marca`: región de origen de la marca (mapeo por diccionario).
- `segmento_marca`: posicionamiento de mercado de la marca (mapeo por diccionario).
- `participacion_mercado`: participación porcentual de cada marca en el dataset.
- `marca_topN`: agrupación Top-N marcas y el resto como *OTRAS*.
- `combustible_norm`: normalización de combustible.
- `transmision_norm`: normalización de transmisión.

> Nota: En este notebook **no** se crea `premium_flag`.
"""
)

with st.expander("Ver fragmentos clave de código"):
    st.code(
        """
# marca_norm + correcciones
df["marca_norm"] = df["marca"].astype(str).str.strip().str.upper().replace("NAN", np.nan)
df["marca_norm"] = df["marca_norm"].replace(brand_corrections)

# participación de mercado (%)
marca_counts = df["marca_norm"].value_counts()
marca_participacion = (marca_counts / len(df) * 100).round(2)
df["participacion_mercado"] = df["marca_norm"].map(marca_participacion)

# Top N marcas
top_brands = set(df["marca_norm"].value_counts().head(top_n).index)
df["marca_topN"] = df["marca_norm"].where(df["marca_norm"].isin(top_brands), "OTRAS")
        """,
        language="python",
    )

# =========================
# Ejecutar FE
# =========================
st.subheader("Aplicar ingeniería de variables")

try:
    df_enriched = feature_engineering(df_raw, top_n=top_n)
except Exception as e:
    st.error(f"Error aplicando feature engineering: {e}")
    st.stop()

new_cols = [
    "marca_norm",
    "origen_marca",
    "segmento_marca",
    "participacion_mercado",
    "marca_topN",
    "combustible_norm",
    "transmision_norm",
]

c1, c2, c3 = st.columns(3)
c1.metric("Filas", "{:,}".format(df_enriched.shape[0]))
c2.metric("Columnas", "{:,}".format(df_enriched.shape[1]))
c3.metric("Variables nuevas", "{}".format(len(new_cols)))

st.markdown("### Vista previa (original + nuevas variables)")
preview_candidates = [
    "marca", "modelo",
    "marca_norm", "origen_marca", "segmento_marca",
    "participacion_mercado", "marca_topN",
    "combustible_norm", "transmision_norm",
]
cols_preview = [c for c in preview_candidates if c in df_enriched.columns]
st.dataframe(df_enriched[cols_preview].head(20), use_container_width=True)

st.markdown("### Resumen rápido de calidad (missing %)")
missing = df_enriched[new_cols].isna().mean().sort_values(ascending=False) * 100
st.dataframe(missing.to_frame("missing_%").round(2), use_container_width=True)

# =========================
# Descargar CSV enriched
# =========================
st.subheader("Descargar dataset enriquecido")

csv_bytes = df_enriched.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar Unsupervised_Learning_enriched.csv",
    data=csv_bytes,
    file_name="Unsupervised_Learning_enriched.csv",
    mime="text/csv",
)
