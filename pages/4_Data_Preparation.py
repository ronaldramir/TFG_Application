import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Preparación de Datos", page_icon="🧼", layout="wide")

# ============================================================
# Funciones (replican Data_Preparation.ipynb)
# ============================================================

def limpiar_precio(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor)
    s = re.sub(r"[¢₡$\(\)\*,]", "", s)
    s = s.strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def limpiar_kilometraje(valor):
    if pd.isna(valor):
        return np.nan

    s = str(valor).lower().strip()

    if s in ["nd", "n/d", "no disponible"]:
        return np.nan

    numero = re.findall(r"[\d,]+", s)
    if not numero:
        return np.nan

    numero = float(numero[0].replace(",", ""))

    if "milla" in s:
        return numero * 1.60934

    if "km" in s:
        return numero

    return numero


def limpiar_cilindrada(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).lower()
    numero = re.findall(r"[\d,]+", s)
    if not numero:
        return np.nan
    return int(numero[0].replace(",", ""))


def corregir_inversion_moneda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrección determinística de inversión CRC/USD (sin conversiones).
    Usa evidencia clara con símbolos en texto:
      - precio_crc_texto contiene '$'
      - precio_usd_texto contiene '¢/₡'
    """
    df = df.copy()

    for col in ["precio_crc_texto", "precio_usd_texto"]:
        if col not in df.columns:
            df[col] = np.nan

    crc_text = df["precio_crc_texto"].astype(str)
    usd_text = df["precio_usd_texto"].astype(str)

    crc_text_has_colon = crc_text.str.contains(r"[¢₡]", na=False)
    crc_text_has_dolar = crc_text.str.contains(r"\$", na=False)

    usd_text_has_colon = usd_text.str.contains(r"[¢₡]", na=False)
    usd_text_has_dolar = usd_text.str.contains(r"\$", na=False)

    swap_mask = crc_text_has_dolar & usd_text_has_colon

    df["precio_crc_fix"] = df.get("precio_crc", np.nan)
    df["precio_usd_fix"] = df.get("precio_usd", np.nan)

    if "precio_crc" in df.columns and "precio_usd" in df.columns:
        df.loc[swap_mask, "precio_crc_fix"] = df.loc[swap_mask, "precio_usd"]
        df.loc[swap_mask, "precio_usd_fix"] = df.loc[swap_mask, "precio_crc"]

        df["precio_crc"] = df["precio_crc_fix"]
        df["precio_usd"] = df["precio_usd_fix"]

    return df


def aplicar_capping(df_in: pd.DataFrame, rules: dict, enable: bool = False):
    if not enable:
        return df_in.copy(), pd.DataFrame(columns=["columna", "low", "high", "cambios"])

    df_out = df_in.copy()
    log = []
    for col, (p_low, p_high) in rules.items():
        if col not in df_out.columns:
            continue
        s = pd.to_numeric(df_out[col], errors="coerce")
        low = s.quantile(p_low)
        high = s.quantile(p_high)
        before = s.copy()
        df_out[col] = s.clip(lower=low, upper=high).round(0).astype("Int64")
        cambios = int((before != df_out[col]).sum(skipna=True))
        log.append((col, float(low), float(high), cambios))

    log_df = pd.DataFrame(log, columns=["columna", "low", "high", "cambios"]).sort_values("cambios", ascending=False)
    return df_out, log_df


def limpiar_y_generar_clean(
    df_raw: pd.DataFrame,
    anio_referencia: int = 2026,
    enable_capping: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Parte 1 del notebook:
    - swap CRC/USD
    - normalizar precios numéricos
    - normalizar kilometraje a km
    - antiguedad = anio_referencia - ano (y se elimina ano)
    - cilindrada a Int
    - pasajeros a Int
    - capping opcional
    - drop columnas no útiles
    """
    df = df_raw.copy()

    # 1) Swap moneda
    df = corregir_inversion_moneda(df)

    # 2) Precios numéricos
    if "precio_crc" in df.columns:
        df["precio_crc"] = df["precio_crc"].apply(limpiar_precio).round(0).astype("Int64")
    if "precio_usd" in df.columns:
        df["precio_usd"] = df["precio_usd"].apply(limpiar_precio).round(0).astype("Int64")

    # 3) Kilometraje
    if "kilometraje" in df.columns:
        df["kilometraje"] = df["kilometraje"].apply(limpiar_kilometraje).round(0).astype("Int64")

    # 4) Antigüedad y drop año
    if "ano" in df.columns:
        df["antiguedad"] = anio_referencia - pd.to_numeric(df["ano"], errors="coerce")
        df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan
        df = df.drop(columns=["ano"], errors="ignore")

    # 5) Cilindrada
    if "cilindrada" in df.columns:
        df["cilindrada"] = df["cilindrada"].apply(limpiar_cilindrada).astype("Int64")

    # 6) Pasajeros
    if "pasajeros" in df.columns:
        df["pasajeros"] = pd.to_numeric(df["pasajeros"], errors="coerce").astype("Int64")

    # 6.1) Capping opcional
    capping_rules = {
        "precio_crc": (0.001, 0.999),
        "kilometraje": (0.001, 0.999),
        "cilindrada": (0.001, 0.999),
        "antiguedad": (0.001, 0.999),
    }
    df, cap_log = aplicar_capping(df, capping_rules, enable=enable_capping)

    # 7) Drop columnas
    columnas_a_borrar = [
        "fecha_ingreso", "visto_texto", "visto_veces", "comentario",
        "car_id", "detail_url", "pagina", "posicion_en_pagina", "source_file",
        "precio_crc_fix", "precio_usd_fix",
        "precio_crc_texto", "precio_usd_texto",
        "titulo_header", "placa"
    ]
    df = df.drop(columns=columnas_a_borrar, errors="ignore")

    meta = {
        "cap_log": cap_log,
    }
    return df, meta


def enriquecer_marcas_y_participacion(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parte 2 del notebook:
    - marca_norm
    - origen_marca
    - segmento_marca
    - participacion_mercado (0–1)
    - tabla de participación por marca en %
    """
    df = df_clean.copy()

    if "marca" not in df.columns:
        return df, pd.DataFrame()

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

    brand_origin_map = {
        # --- JAPÓN ---
        "TOYOTA": "JAPON", "HONDA": "JAPON", "NISSAN": "JAPON", "MAZDA": "JAPON",
        "MITSUBISHI": "JAPON", "SUZUKI": "JAPON", "SUBARU": "JAPON", "LEXUS": "JAPON",
        # --- COREA ---
        "HYUNDAI": "COREA", "KIA": "COREA", "SSANGYONG": "COREA",
        # --- ALEMANIA ---
        "BMW": "ALEMANIA", "AUDI": "ALEMANIA", "MERCEDES-BENZ": "ALEMANIA", "VOLKSWAGEN": "ALEMANIA", "PORSCHE": "ALEMANIA",
        # --- USA ---
        "FORD": "USA", "CHEVROLET": "USA", "JEEP": "USA", "DODGE": "USA", "TESLA": "USA",
        # --- FRANCIA ---
        "PEUGEOT": "FRANCIA", "RENAULT": "FRANCIA", "CITROEN": "FRANCIA",
        # --- CHINA (ejemplos) ---
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

    marca_counts = df["marca_norm"].value_counts(dropna=False)
    total_autos = len(df)
    df["participacion_mercado"] = df["marca_norm"].map(marca_counts / total_autos)

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

    participacion_df["participacion_pct"] = (participacion_df["conteo"] / total_autos * 100).round(2)
    participacion_df = participacion_df.sort_values("participacion_pct", ascending=False)
    participacion_df = participacion_df.drop(columns=["conteo"])

    return df, participacion_df


# ============================================================
# UI
# ============================================================

st.title("🧼 Preparación de Datos (Data_Preparation.ipynb)")
st.caption("Esta página replica el flujo del notebook final: **CLEAN** → **ENRICHED** + CSV de participación por marca.")

with st.container(border=True):
    st.subheader("Qué se hace en este bloque")
    st.markdown(
        """
**Parte A (CLEAN):**
- Detecta swaps CRC/USD usando evidencia de símbolos en texto y corrige sin conversiones.
- Limpia `precio_crc` y `precio_usd` para dejarlos numéricos.
- Convierte `kilometraje` a km (si viene en millas, se convierte).
- Crea `antiguedad` con año de referencia (y elimina `ano`).
- Limpia `cilindrada` y `pasajeros`.
- (Opcional) capping por cuantiles.
- Elimina columnas que no aportan al clustering/modelado.

**Parte B (ENRICHED):**
- `marca_norm`, `origen_marca`, `segmento_marca`.
- `participacion_mercado` (0–1) por marca.
- CSV adicional de **participación por marca en %**.
        """
    )

st.divider()

tab1, tab2 = st.tabs(["A) Generar CLEAN", "B) Enriquecer (ENRICHED)"])

# -------------------------
# TAB A
# -------------------------
with tab1:
    st.subheader("Generar CR_Autos_Cleaned (CLEAN)")
    st.markdown("Sube el dataset original (por ejemplo `crautos_merge_final.csv`) y aplica el pipeline CLEAN.")

    colA1, colA2, colA3 = st.columns([1,1,2])
    with colA1:
        anio_ref = st.number_input("Año de referencia (antigüedad)", min_value=2000, max_value=2100, value=2026, step=1)
    with colA2:
        enable_capping = st.toggle("Capping (opcional)", value=False)
    with colA3:
        st.caption("Si no quieres capping, déjalo apagado (como en tu notebook).")

    up = st.file_uploader("Subir CSV original", type=["csv"], key="uploader_clean")
    if up is not None:
        df_raw = pd.read_csv(up)
        st.write("Vista previa (raw):")
        st.dataframe(df_raw.head(20), use_container_width=True)

        if st.button("Aplicar CLEAN", type="primary"):
            df_clean, meta = limpiar_y_generar_clean(df_raw, anio_referencia=int(anio_ref), enable_capping=enable_capping)

            st.success("CLEAN generado.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Filas", df_clean.shape[0])
            with c2:
                st.metric("Columnas", df_clean.shape[1])
            with c3:
                st.metric("Nulos (total)", int(df_clean.isna().sum().sum()))

            st.write("Vista previa (CLEAN):")
            st.dataframe(df_clean.head(25), use_container_width=True)

            if enable_capping and not meta["cap_log"].empty:
                st.markdown("**Log de capping (cuántos valores fueron recortados):**")
                st.dataframe(meta["cap_log"], use_container_width=True)

            csv_clean = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CR_Autos_Cleaned.csv",
                data=csv_clean,
                file_name="CR_Autos_Cleaned.csv",
                mime="text/csv"
            )

# -------------------------
# TAB B
# -------------------------
with tab2:
    st.subheader("Generar CR_Autos_Cleaned_enriched (ENRICHED)")
    st.markdown("Sube el dataset CLEAN (`CR_Autos_Cleaned.csv`) y aplica el enriquecimiento de marca.")

    up2 = st.file_uploader("Subir CSV CLEAN", type=["csv"], key="uploader_enrich")
    if up2 is not None:
        df_clean_in = pd.read_csv(up2)
        st.write("Vista previa (CLEAN):")
        st.dataframe(df_clean_in.head(20), use_container_width=True)

        if st.button("Aplicar ENRICHED", type="primary"):
            df_enriched, part_df = enriquecer_marcas_y_participacion(df_clean_in)

            st.success("ENRICHED generado.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Filas", df_enriched.shape[0])
            with c2:
                st.metric("Columnas", df_enriched.shape[1])
            with c3:
                st.metric("Marcas únicas", int(df_enriched["marca_norm"].nunique(dropna=False)) if "marca_norm" in df_enriched.columns else 0)

            st.write("Vista previa (ENRICHED):")
            st.dataframe(df_enriched.head(25), use_container_width=True)

            if not part_df.empty:
                st.markdown("**Participación de mercado por marca (%):**")
                st.dataframe(part_df.head(25), use_container_width=True)

            csv_enriched = df_enriched.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CR_Autos_Cleaned_enriched.csv",
                data=csv_enriched,
                file_name="CR_Autos_Cleaned_enriched.csv",
                mime="text/csv"
            )

            if not part_df.empty:
                csv_part = part_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar participacion_mercado_marcas.csv",
                    data=csv_part,
                    file_name="participacion_mercado_marcas.csv",
                    mime="text/csv"
                )

st.divider()

with st.expander("Snippets del notebook (para documentación)"):
    st.markdown("**Swap CRC/USD (evidencia por símbolos):**")
    st.code(
        """swap_mask = crc_text_has_dolar & usd_text_has_colon
df.loc[swap_mask, "precio_crc_fix"] = df.loc[swap_mask, "precio_usd"]
df.loc[swap_mask, "precio_usd_fix"] = df.loc[swap_mask, "precio_crc"]""",
        language="python"
    )

    st.markdown("**Antigüedad con año de referencia:**")
    st.code(
        """ANIO_REFERENCIA = 2026
df["antiguedad"] = ANIO_REFERENCIA - df["ano"]
df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan
df = df.drop(columns=["ano"], errors="ignore")""",
        language="python"
    )

    st.markdown("**Participación de mercado (0–1) + CSV en %:**")
    st.code(
        """marca_counts = df["marca_norm"].value_counts(dropna=False)
total_autos = len(df)
df["participacion_mercado"] = df["marca_norm"].map(marca_counts / total_autos)

participacion_df["participacion_pct"] = (
    participacion_df["conteo"] / total_autos * 100
).round(2)""",
        language="python"
    )
