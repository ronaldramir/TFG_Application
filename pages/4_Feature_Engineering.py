import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Preparación de Datos",
    page_icon="🧼",
    layout="wide"
)

# ============================================================
# Helpers (alineados al notebook Data_Preparation.ipynb)
# ============================================================

def normalizar_marca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "marca" not in df.columns:
        return df

    df["marca_norm"] = (
        df["marca"]
        .astype(str)
        .str.strip()
        .str.upper()
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
    return df


def enriquecer_origen_segmento(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "marca_norm" not in df.columns:
        return df

    brand_origin_map = {
        # JAPÓN
        "TOYOTA": "JAPON", "HONDA": "JAPON", "NISSAN": "JAPON", "MAZDA": "JAPON",
        "MITSUBISHI": "JAPON", "SUZUKI": "JAPON", "SUBARU": "JAPON", "LEXUS": "JAPON",
        # COREA
        "HYUNDAI": "COREA", "KIA": "COREA", "SSANGYONG": "COREA",
        # ALEMANIA
        "BMW": "ALEMANIA", "AUDI": "ALEMANIA", "MERCEDES-BENZ": "ALEMANIA",
        "VOLKSWAGEN": "ALEMANIA", "PORSCHE": "ALEMANIA",
        # USA
        "FORD": "USA", "CHEVROLET": "USA", "JEEP": "USA", "DODGE": "USA", "TESLA": "USA",
        # FRANCIA
        "PEUGEOT": "FRANCIA", "RENAULT": "FRANCIA", "CITROEN": "FRANCIA",
        # CHINA (ejemplos)
        "CHERY": "CHINA", "GEELY": "CHINA", "BYD": "CHINA", "MG": "CHINA", "GREAT WALL": "CHINA",
    }
    df["origen_marca"] = df["marca_norm"].map(brand_origin_map).fillna("DESCONOCIDO")

    brand_segment_map = {
        # Premium (ejemplos)
        "BMW": "PREMIUM", "AUDI": "PREMIUM", "MERCEDES-BENZ": "PREMIUM", "PORSCHE": "PREMIUM",
        "LAND ROVER": "PREMIUM", "LEXUS": "PREMIUM", "ROLLS-ROYCE": "PREMIUM",
        # Generalista (ejemplos)
        "TOYOTA": "GENERALISTA", "HONDA": "GENERALISTA", "NISSAN": "GENERALISTA",
        "HYUNDAI": "GENERALISTA", "KIA": "GENERALISTA", "MAZDA": "GENERALISTA",
        "SUZUKI": "GENERALISTA", "FORD": "GENERALISTA", "CHEVROLET": "GENERALISTA",
        # Comercial (ejemplos)
        "ISUZU": "COMERCIAL",
    }
    df["segmento_marca"] = df["marca_norm"].map(brand_segment_map).fillna("DESCONOCIDO")
    return df


def crear_participacion_mercado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea participacion_mercado como proporción 0-1 por marca_norm.
    (Esto reemplaza la idea de 'marca_freq' cuando se quiere un feature escalable).
    """
    df = df.copy()
    if "marca_norm" not in df.columns:
        return df

    marca_counts = df["marca_norm"].value_counts(dropna=False)
    total_autos = len(df)
    df["participacion_mercado"] = df["marca_norm"].map(marca_counts / total_autos)
    return df


def pipeline_preparacion(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2 = normalizar_marca(df2)
    df2 = enriquecer_origen_segmento(df2)
    df2 = crear_participacion_mercado(df2)
    return df2


def tabla_participacion(df: pd.DataFrame) -> pd.DataFrame:
    """Genera el CSV 'participacion_mercado_marcas' tal como en el notebook (con %)."""
    if "marca_norm" not in df.columns:
        return pd.DataFrame()

    total_autos = len(df)

    out = (
        df.groupby("marca_norm", dropna=False)
        .agg({
            "participacion_mercado": "first",  # 0-1
            "origen_marca": "first",
            "segmento_marca": "first",
            "marca": "count"
        })
        .rename(columns={"marca": "conteo"})
        .reset_index()
    )

    out["participacion_pct"] = (out["conteo"] / total_autos * 100).round(2)
    out = out.sort_values("participacion_pct", ascending=False)

    # Igual que tu notebook: se elimina el conteo si no se quiere en el CSV final
    out = out.drop(columns=["conteo"])
    return out


# ============================================================
# UI
# ============================================================

st.title("🧼 Preparación de Datos")
st.caption("Página alineada con el notebook `Data_Preparation.ipynb`: enriquecimiento de marca + participación de mercado.")

with st.container(border=True):
    st.subheader("¿Qué se hizo aquí?")
    st.markdown(
        """
En esta fase se crean variables derivadas **centradas en la marca**, para apoyar tanto análisis descriptivo
como modelado (clustering / predicción).

Incluye:
- Normalización de `marca` → `marca_norm` (limpieza de texto + alias).
- Derivación de `origen_marca` (mapeo) y `segmento_marca` (mapeo).
- Cálculo de `participacion_mercado` como proporción 0–1 por marca (reemplaza “frecuencia” cuando se prefiere una escala relativa).
- Generación opcional de un CSV adicional: `participacion_mercado_marcas.csv` con `%` ya listo para reportar.
        """
    )

st.divider()

# ============================================================
# DEMO INTERACTIVO
# ============================================================

with st.container(border=True):
    st.subheader("Demo rápido")
    st.markdown("Sube un CSV (por ejemplo `CR_Autos_Cleaned.csv`) y aplica el pipeline del notebook.")

    up = st.file_uploader("Subir CSV", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
        st.write("Vista previa (raw):")
        st.dataframe(df_raw.head(20), use_container_width=True)

        if st.button("Aplicar preparación", type="primary"):
            df_prep = pipeline_preparacion(df_raw)

            st.success("Listo. Dataset enriquecido.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Filas", df_prep.shape[0])
            with c2:
                st.metric("Columnas", df_prep.shape[1])
            with c3:
                st.metric("Marcas únicas", int(df_prep["marca_norm"].nunique(dropna=False)) if "marca_norm" in df_prep.columns else 0)

            st.write("Vista previa (enriched):")
            st.dataframe(df_prep.head(20), use_container_width=True)

            # Tabla de participación de mercado por marca (%)
            part_df = tabla_participacion(df_prep)
            if not part_df.empty:
                st.markdown("**Participación de mercado por marca (%):**")
                st.dataframe(part_df.head(25), use_container_width=True)

            # Descargas
            csv_enriched = df_prep.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CSV enriquecido",
                data=csv_enriched,
                file_name="CR_Autos_Cleaned_enriched.csv",
                mime="text/csv"
            )

            if not part_df.empty:
                csv_part = part_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar participación por marca (CSV)",
                    data=csv_part,
                    file_name="participacion_mercado_marcas.csv",
                    mime="text/csv"
                )

st.divider()

with st.expander("Ver snippet del cálculo de participación (como en el notebook)"):
    st.code(
        """
marca_counts = df["marca_norm"].value_counts(dropna=False)
total_autos = len(df)
df["participacion_mercado"] = df["marca_norm"].map(marca_counts / total_autos)
        """,
        language="python"
    )

with st.expander("Ver snippet del CSV de participación en %"):
    st.code(
        """
participacion_df = (
    df.groupby("marca_norm")
      .agg({
          "participacion_mercado": "first",
          "origen_marca": "first",
          "segmento_marca": "first",
          "marca": "count"
      })
      .rename(columns={"marca": "conteo"})
      .reset_index()
)

participacion_df["participacion_pct"] = (
    participacion_df["conteo"] / total_autos * 100
).round(2)

participacion_df = participacion_df.sort_values("participacion_pct", ascending=False)
participacion_df = participacion_df.drop(columns=["conteo"])
        """,
        language="python"
    )

st.info("Si luego quieres extender esta página con limpieza numérica u otras features, se agrega encima sin drama.")
