import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(
    page_title="Preparación de Datos",
    page_icon="🧼",
    layout="wide"
)

# ============================================================
# Helpers (replican la lógica del notebook Data_Preparation)
# ============================================================

def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convierte texto numérico con separadores y símbolos a número (float)."""
    if series is None:
        return series
    s = series.astype(str).str.replace(r"[^\d\.]", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def corregir_inversion_moneda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrección determinística de inversión CRC/USD sin conversiones:
    - Usa columnas de texto (precio_crc_texto, precio_usd_texto) si existen
    - Detecta símbolos ¢/₡ y $ y corrige si están invertidos
    - Crea precio_crc_fix y precio_usd_fix y luego actualiza precio_crc/precio_usd
    """
    df = df.copy()

    for col in ["precio_crc_texto", "precio_usd_texto"]:
        if col not in df.columns:
            df[col] = np.nan

    # Asegurar numéricos (si vienen como texto)
    for col in ["precio_crc", "precio_usd"]:
        if col in df.columns:
            df[col] = _to_numeric_safe(df[col])

    crc_text = df["precio_crc_texto"].astype(str)
    usd_text = df["precio_usd_texto"].astype(str)

    crc_has_colon = crc_text.str.contains(r"[¢₡]", na=False)
    crc_has_dolar = crc_text.str.contains(r"\$", na=False)

    usd_has_colon = usd_text.str.contains(r"[¢₡]", na=False)
    usd_has_dolar = usd_text.str.contains(r"\$", na=False)

    # Caso típico de inversión:
    # - precio_crc_texto trae '$' (parece USD)
    # - precio_usd_texto trae '¢/₡' (parece CRC)
    invertido = (crc_has_dolar & usd_has_colon)

    df["precio_crc_fix"] = df.get("precio_crc", np.nan)
    df["precio_usd_fix"] = df.get("precio_usd", np.nan)

    if "precio_crc" in df.columns and "precio_usd" in df.columns:
        df.loc[invertido, "precio_crc_fix"] = df.loc[invertido, "precio_usd"]
        df.loc[invertido, "precio_usd_fix"] = df.loc[invertido, "precio_crc"]

        # Sustituir columnas principales por las corregidas
        df["precio_crc"] = df["precio_crc_fix"]
        df["precio_usd"] = df["precio_usd_fix"]

    return df


def crear_antiguedad(df: pd.DataFrame, col_ano: str = "ano") -> pd.DataFrame:
    """Crea antiguedad = año_actual - año del vehículo."""
    df = df.copy()
    if col_ano in df.columns:
        ano = _to_numeric_safe(df[col_ano])
        current_year = datetime.now().year
        df["antiguedad"] = (current_year - ano).astype("Float64")
    return df


def enriquecer_marcas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la ingeniería de marcas del notebook:
    - marca_norm (upper/strip + alias)
    - origen_marca (mapeo por región)
    - segmento_marca (PREMIUM/ECONOMICO/COMERCIAL/GENERAL)
    - premium_flag (segmento_marca == PREMIUM)
    - marca_freq (frecuencia absoluta por marca_norm)
    - marca_topN (Top-N vs OTRAS)
    """
    df = df.copy()
    if "marca" not in df.columns:
        return df

    df["marca_norm"] = (
        df["marca"].astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )

    marca_alias = {
        "DONFENG": "DONGFENG",
        "SSANG": "SSANGYONG",
        "GREAT": "GREAT WALL",
        "DODGE/RAM": "DODGE",
        "LAND": "LAND ROVER",
        "ROVER": "LAND ROVER",
        "ROLLS": "ROLLS-ROYCE",
    }
    df["marca_norm"] = df["marca_norm"].replace(marca_alias)

    # --- Origen de marca (resumen, pero alineado a la idea del notebook) ---
    brand_origin_map = {
        # JAPÓN
        "TOYOTA": "JAPON", "HONDA": "JAPON", "NISSAN": "JAPON", "MAZDA": "JAPON",
        "MITSUBISHI": "JAPON", "SUBARU": "JAPON", "SUZUKI": "JAPON", "DAIHATSU": "JAPON",
        "ISUZU": "JAPON", "LEXUS": "JAPON", "INFINITI": "JAPON",
        # COREA
        "HYUNDAI": "COREA", "KIA": "COREA", "SSANGYONG": "COREA", "GENESIS": "COREA",
        # USA
        "FORD": "USA", "CHEVROLET": "USA", "GMC": "USA", "DODGE": "USA", "JEEP": "USA",
        "CHRYSLER": "USA", "TESLA": "USA", "CADILLAC": "USA",
        # EUROPA
        "BMW": "EUROPA", "MERCEDES": "EUROPA", "AUDI": "EUROPA", "VOLKSWAGEN": "EUROPA",
        "VOLVO": "EUROPA", "PORSCHE": "EUROPA", "LAND ROVER": "EUROPA", "JAGUAR": "EUROPA",
        "MINI": "EUROPA", "ALFA": "EUROPA", "FIAT": "EUROPA", "PEUGEOT": "EUROPA",
        "RENAULT": "EUROPA", "SEAT": "EUROPA", "SKODA": "EUROPA", "OPEL": "EUROPA",
        "BENTLEY": "EUROPA", "FERRARI": "EUROPA", "LAMBORGHINI": "EUROPA",
        "MASERATI": "EUROPA", "ROLLS-ROYCE": "EUROPA",
        # CHINA
        "CHERY": "CHINA", "CHANGAN": "CHINA", "BAIC": "CHINA", "JAC": "CHINA",
        "GEELY": "CHINA", "DONGFENG": "CHINA", "FAW": "CHINA", "SOUEAST": "CHINA",
        "GREAT WALL": "CHINA", "JETOUR": "CHINA", "OMODA": "CHINA", "ZEEKR": "CHINA",
        "XPENG": "CHINA",
        # OTROS
        "TATA": "INDIA",
    }

    df["origen_marca"] = df["marca_norm"].map(brand_origin_map).fillna("OTRO")

    segment_map = {
        # PREMIUM
        "AUDI": "PREMIUM", "BMW": "PREMIUM", "MERCEDES": "PREMIUM", "LEXUS": "PREMIUM",
        "INFINITI": "PREMIUM", "GENESIS": "PREMIUM", "JAGUAR": "PREMIUM", "LAND ROVER": "PREMIUM",
        "PORSCHE": "PREMIUM", "VOLVO": "PREMIUM", "MINI": "PREMIUM", "ALFA": "PREMIUM",
        "BENTLEY": "PREMIUM", "FERRARI": "PREMIUM", "LAMBORGHINI": "PREMIUM", "MASERATI": "PREMIUM",
        "ROLLS-ROYCE": "PREMIUM", "CADILLAC": "PREMIUM", "TESLA": "PREMIUM", "ZEEKR": "PREMIUM", "XPENG": "PREMIUM",
        # ECONÓMICO
        "CHERY": "ECONOMICO", "CHANGAN": "ECONOMICO", "BAIC": "ECONOMICO", "JAC": "ECONOMICO",
        "GEELY": "ECONOMICO", "DONGFENG": "ECONOMICO", "FAW": "ECONOMICO", "SOUEAST": "ECONOMICO",
        "JETOUR": "ECONOMICO", "OMODA": "ECONOMICO",
        # COMERCIAL
        "HINO": "COMERCIAL", "ISUZU": "COMERCIAL", "FOTON": "COMERCIAL",
        "FREIGHTLINER": "COMERCIAL", "INTERNATIONAL": "COMERCIAL", "MACK": "COMERCIAL",
        "PETERBILT": "COMERCIAL",
    }

    df["segmento_marca"] = df["marca_norm"].map(segment_map).fillna("GENERAL")
    df["premium_flag"] = (df["segmento_marca"] == "PREMIUM").astype(int)

    marca_freq = df["marca_norm"].value_counts(dropna=False)
    df["marca_freq"] = df["marca_norm"].map(marca_freq).astype("Int64")

    top_n = 20
    top_brands = set(df["marca_norm"].value_counts().head(top_n).index)
    df["marca_topN"] = np.where(df["marca_norm"].isin(top_brands), df["marca_norm"], "OTRAS")

    return df


def normalizar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza combustible y transmisión (upper/strip y limpieza simple)."""
    df = df.copy()

    if "combustible" in df.columns:
        df["combustible_norm"] = (
            df["combustible"].astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )
        df["combustible_norm"] = df["combustible_norm"].replace({
            "ELÉCTRICO": "ELECTRICO",
            "HÍBRIDO": "HIBRIDO",
        })

    if "transmision" in df.columns:
        df["transmision_norm"] = (
            df["transmision"].astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )
        df["transmision_norm"] = df["transmision_norm"].replace({
            "AUTOMATICA": "AUTOMATICA",
            "AUTOMÁTICA": "AUTOMATICA",
            "MANUAL": "MANUAL",
        })

    return df


def pipeline_preparacion(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de preparación (según el notebook)."""
    df2 = df.copy()
    df2 = corregir_inversion_moneda(df2)
    df2 = crear_antiguedad(df2, col_ano="ano")
    df2 = enriquecer_marcas(df2)
    df2 = normalizar_categoricas(df2)
    return df2


# ============================================================
# UI
# ============================================================

st.title("🧼 Preparación de Datos")
st.caption("Página actualizada a partir del notebook `Data_Preparation.ipynb` (limpieza + variables derivadas).")

with st.container(border=True):
    st.subheader("¿Qué se hizo en esta fase?")
    st.markdown(
        """
Esta etapa convierte el dataset “crudo” (scraping) o el dataset “base” (Unsupervised_Learning.csv)
en un dataset más consistente para análisis y modelado.

Incluye principalmente:
- Corrección determinística de moneda CRC/USD cuando el scraping invierte campos (sin conversiones).
- Tipificación y limpieza de variables numéricas relevantes.
- Derivación de `antiguedad`.
- Estandarización y enriquecimiento de marca: `marca_norm`, `origen_marca`, `segmento_marca`, `premium_flag`,
  `marca_freq` y `marca_topN`.
- Normalización de `combustible` y `transmision` para evitar duplicados por texto.
        """
    )

st.divider()

# ============================================================
# DEMO INTERACTIVO
# ============================================================

with st.container(border=True):
    st.subheader("Demo rápido (opcional)")
    st.markdown("Sube un CSV y aplica el pipeline de preparación para ver el efecto y descargar el resultado.")

    up = st.file_uploader("Subir CSV", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
        st.write("Vista previa (raw):")
        st.dataframe(df_raw.head(20), use_container_width=True)

        if st.button("Aplicar preparación", type="primary"):
            df_prep = pipeline_preparacion(df_raw)

            st.success("Listo. Dataset preparado.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Filas", df_prep.shape[0])
            with c2:
                st.metric("Columnas", df_prep.shape[1])
            with c3:
                n_premium = int(df_prep.get("premium_flag", pd.Series([0]*len(df_prep))).sum()) if len(df_prep) else 0
                st.metric("Premium (flag=1)", n_premium)

            st.write("Vista previa (prepared):")
            st.dataframe(df_prep.head(20), use_container_width=True)

            # Frecuencia de marca como %
            if "marca_norm" in df_prep.columns:
                st.markdown("**Top marcas (porcentaje del dataset):**")
                vc = df_prep["marca_norm"].value_counts(dropna=False)
                pct = (vc / vc.sum() * 100).round(2).rename("porcentaje")
                tabla = pct.reset_index().rename(columns={"index": "marca_norm"}).head(20)
                st.dataframe(tabla, use_container_width=True)

            # Descargar
            csv_bytes = df_prep.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CSV preparado",
                data=csv_bytes,
                file_name="dataset_preparado.csv",
                mime="text/csv"
            )

st.divider()

# ============================================================
# DOCUMENTACIÓN (explicación + snippets)
# ============================================================

with st.expander("Ver lógica de corrección de moneda (CRC/USD)"):
    st.markdown(
        """
**Problema observado:** durante el scraping, algunos anuncios muestran el monto correcto,
pero el campo termina en la columna equivocada (CRC vs USD).

**Solución aplicada (sin conversiones):**
- Se leen columnas texto (si existen): `precio_crc_texto`, `precio_usd_texto`.
- Si `precio_crc_texto` contiene `$` y `precio_usd_texto` contiene `¢/₡`, se asume inversión.
- Se intercambian los valores numéricos y se preserva el resultado en `precio_crc_fix` y `precio_usd_fix`.
        """
    )
    st.code(
        """
invertido = (crc_has_dolar & usd_has_colon)
df.loc[invertido, "precio_crc_fix"] = df.loc[invertido, "precio_usd"]
df.loc[invertido, "precio_usd_fix"] = df.loc[invertido, "precio_crc"]
        """,
        language="python"
    )

with st.expander("Ver lógica de antigüedad"):
    st.markdown("Se crea `antiguedad = año_actual - ano` para capturar depreciación de manera más directa.")
    st.code(
        """
current_year = datetime.now().year
df["antiguedad"] = current_year - pd.to_numeric(df["ano"], errors="coerce")
        """,
        language="python"
    )

with st.expander("Ver lógica de marcas (normalización + variables derivadas)"):
    st.markdown(
        """
Se normaliza `marca` para reducir inconsistencias, luego se derivan variables de dominio:
- `origen_marca`: región de origen (mapeo).
- `segmento_marca`: PREMIUM / ECONOMICO / COMERCIAL / GENERAL (mapeo).
- `premium_flag`: indicador binario.
- `marca_freq`: frecuencia absoluta (útil como numérica sin one-hot).
- `marca_topN`: reduce cardinalidad a Top-20 y el resto como `OTRAS`.
        """
    )

    st.code(
        """
df["marca_norm"] = df["marca"].astype(str).str.strip().str.upper()
df["origen_marca"] = df["marca_norm"].map(brand_origin_map).fillna("OTRO")
df["segmento_marca"] = df["marca_norm"].map(segment_map).fillna("GENERAL")
df["premium_flag"] = (df["segmento_marca"] == "PREMIUM").astype(int)

marca_freq = df["marca_norm"].value_counts(dropna=False)
df["marca_freq"] = df["marca_norm"].map(marca_freq).astype("Int64")

top_n = 20
top_brands = set(df["marca_norm"].value_counts().head(top_n).index)
df["marca_topN"] = np.where(df["marca_norm"].isin(top_brands), df["marca_norm"], "OTRAS")
        """,
        language="python"
    )

with st.expander("Ver normalización de combustible y transmisión"):
    st.code(
        """
df["combustible_norm"] = df["combustible"].astype(str).str.strip().str.upper()
df["transmision_norm"] = df["transmision"].astype(str).str.strip().str.upper()
        """,
        language="python"
    )

st.info(
    "Nota: el escalado (StandardScaler) y el preprocesamiento para modelado se aplican en la fase de modelado "
    "(pipelines), no necesariamente en esta etapa."
)
