"""
feature_engineering.py
------------------------------------------------------------
Feature Engineering para el proyecto de autos usados (Costa Rica)

Este módulo genera el dataset "enriched" a partir del CSV original,
agregando las siguientes columnas:

  - marca_norm     : marca normalizada (MAYÚSCULAS, sin tildes, sin espacios extra) + alias
  - origen_marca   : región de origen de la marca (lookup por diccionario)
  - segmento_marca : segmento de mercado de la marca (lookup por diccionario)
  - premium_flag   : 1 si segmento_marca == "PREMIUM", si no 0
  - marca_freq     : frecuencia de aparición de marca_norm en el dataset
  - marca_topN     : Top N marcas más frecuentes; el resto -> "OTRAS"

Uso típico desde Streamlit:
    from feature_engineering import build_enriched_features, transform_single_input, FEConfig

    df = pd.read_csv("Unsupervised_Learning.csv")
    df_enriched = build_enriched_features(df)

    # Para un registro ingresado por el usuario:
    user_row = transform_single_input(
        {"marca": "Toyota", "modelo": "RAV4", "anio": 2018, ...},
        df_reference_for_freq=df_enriched  # o el df original ya enriquecido
    )

Uso como script (genera CSV):
    python feature_engineering.py --in Unsupervised_Learning.csv --out Unsupervised_Learning_enriched.csv --topn 20
------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =========================
# Configuración
# =========================

@dataclass(frozen=True)
class FEConfig:
    """Configuración del feature engineering."""
    top_n_brands: int = 20
    unknown_origin: str = "OTROS"   # default usado si no hay mapeo
    unknown_segment: str = "MEDIO" # default usado si no hay mapeo


# =========================
# Diccionarios (ajustables)
# =========================
# NOTA: Estos diccionarios pueden/ deben ajustarse a tu proyecto.
# Los defaults aquí son razonables, pero si ya tienes los tuyos exactos,
# reemplázalos por los que usaste en tu notebook.

MARCA_ALIAS: Dict[str, str] = {
    # correcciones frecuentes por scraping / escritura
    "DONFENG": "DONGFENG",
    "SSANG": "SSANGYONG",
    "GREAT": "GREAT WALL",
    "DODGE/RAM": "DODGE",
    "LAND": "LAND ROVER",
    "ROVER": "LAND ROVER",
    "ROLLS": "ROLLS-ROYCE",
}

BRAND_ORIGIN_MAP: Dict[str, str] = {
    # JAPÓN
    "TOYOTA": "JAPON",
    "HONDA": "JAPON",
    "NISSAN": "JAPON",
    "MAZDA": "JAPON",
    "MITSUBISHI": "JAPON",
    "SUBARU": "JAPON",
    "SUZUKI": "JAPON",
    "LEXUS": "JAPON",
    "INFINITI": "JAPON",

    # COREA
    "KIA": "COREA",
    "HYUNDAI": "COREA",
    "SSANGYONG": "COREA",
    "GENESIS": "COREA",

    # USA
    "FORD": "USA",
    "CHEVROLET": "USA",
    "JEEP": "USA",
    "DODGE": "USA",
    "CHRYSLER": "USA",
    "GMC": "USA",
    "TESLA": "USA",

    # EUROPA
    "BMW": "EUROPA",
    "MERCEDES-BENZ": "EUROPA",
    "AUDI": "EUROPA",
    "VOLKSWAGEN": "EUROPA",
    "VOLVO": "EUROPA",
    "PEUGEOT": "EUROPA",
    "RENAULT": "EUROPA",
    "LAND ROVER": "EUROPA",
    "PORSCHE": "EUROPA",
    "MINI": "EUROPA",

    # CHINA
    "GEELY": "CHINA",
    "CHERY": "CHINA",
    "GREAT WALL": "CHINA",
    "DONGFENG": "CHINA",
}

BRAND_SEGMENT_MAP: Dict[str, str] = {
    # PREMIUM
    "BMW": "PREMIUM",
    "MERCEDES-BENZ": "PREMIUM",
    "AUDI": "PREMIUM",
    "LEXUS": "PREMIUM",
    "PORSCHE": "PREMIUM",
    "LAND ROVER": "PREMIUM",
    "VOLVO": "PREMIUM",
    "INFINITI": "PREMIUM",

    # MEDIO
    "TOYOTA": "MEDIO",
    "HONDA": "MEDIO",
    "NISSAN": "MEDIO",
    "MAZDA": "MEDIO",
    "SUBARU": "MEDIO",
    "VOLKSWAGEN": "MEDIO",
    "FORD": "MEDIO",
    "CHEVROLET": "MEDIO",
    "JEEP": "MEDIO",

    # ECONOMICO
    "KIA": "ECONOMICO",
    "HYUNDAI": "ECONOMICO",
    "SUZUKI": "ECONOMICO",
    "CHERY": "ECONOMICO",
    "GEELY": "ECONOMICO",
    "GREAT WALL": "ECONOMICO",
    "DONGFENG": "ECONOMICO",

    # Si manejas "COMERCIAL" u otros, agrégalos aquí
}


# =========================
# Helpers de normalización
# =========================

def _strip_accents(text: str) -> str:
    """Quita tildes/acentos para normalizar strings."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def normalize_brand(x: Any) -> Optional[str]:
    """
    Normaliza marca:
      - convierte a string
      - trim
      - MAYÚSCULAS
      - sin tildes
      - filtra valores vacíos / nulos típicos
    """
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = _strip_accents(s)

    if s in {"", "NAN", "NONE", "NULL"}:
        return None
    return s

def apply_brand_alias(marca_norm: Optional[str]) -> Optional[str]:
    """Unifica marcas con alias (scraping)."""
    if marca_norm is None:
        return None
    return MARCA_ALIAS.get(marca_norm, marca_norm)


# =========================
# Transformaciones
# =========================

def build_enriched_features(
    df: pd.DataFrame,
    config: FEConfig = FEConfig(),
    origin_map: Dict[str, str] = BRAND_ORIGIN_MAP,
    segment_map: Dict[str, str] = BRAND_SEGMENT_MAP,
    marca_col: str = "marca",
) -> pd.DataFrame:
    """
    Recibe el DataFrame original y devuelve una copia con columnas enriched.

    Requisitos:
      - Debe existir la columna `marca_col` (por default: "marca")
    """
    if marca_col not in df.columns:
        raise KeyError(f"No existe la columna '{marca_col}' en el DataFrame.")

    out = df.copy()

    # 1) marca_norm
    out["marca_norm"] = out[marca_col].apply(normalize_brand).apply(apply_brand_alias)

    # 2) origen_marca (lookup por marca_norm)
    out["origen_marca"] = out["marca_norm"].map(origin_map).fillna(config.unknown_origin)

    # 3) segmento_marca (lookup por marca_norm)
    out["segmento_marca"] = out["marca_norm"].map(segment_map).fillna(config.unknown_segment)

    # 4) premium_flag (binaria)
    out["premium_flag"] = (out["segmento_marca"] == "PREMIUM").astype(int)

    # 5) marca_freq (popularidad numérica)
    freq = out["marca_norm"].value_counts(dropna=False)
    out["marca_freq"] = out["marca_norm"].map(freq).astype("Int64")

    # 6) marca_topN (reducción de cardinalidad)
    top_brands = set(out["marca_norm"].value_counts().head(config.top_n_brands).index)
    out["marca_topN"] = np.where(out["marca_norm"].isin(top_brands), out["marca_norm"], "OTRAS")

    return out


def transform_single_input(
    input_dict: Dict[str, Any],
    df_reference_for_freq: Optional[pd.DataFrame] = None,
    config: FEConfig = FEConfig(),
    origin_map: Dict[str, str] = BRAND_ORIGIN_MAP,
    segment_map: Dict[str, str] = BRAND_SEGMENT_MAP,
    marca_col: str = "marca",
) -> pd.DataFrame:
    """
    Para Streamlit: transforma un solo registro (dict) y devuelve un DataFrame de 1 fila.

    - Si `df_reference_for_freq` se proporciona, calcula marca_freq y marca_topN basándose en ese dataset.
    - Si no, usa defaults razonables: marca_freq=1 y marca_topN="OTRAS".
    """
    row = pd.DataFrame([input_dict])

    if marca_col not in row.columns:
        # no reventamos, pero dejamos marca en None
        row[marca_col] = None

    row["marca_norm"] = row[marca_col].apply(normalize_brand).apply(apply_brand_alias)
    row["origen_marca"] = row["marca_norm"].map(origin_map).fillna(config.unknown_origin)
    row["segmento_marca"] = row["marca_norm"].map(segment_map).fillna(config.unknown_segment)
    row["premium_flag"] = (row["segmento_marca"] == "PREMIUM").astype(int)

    if df_reference_for_freq is not None and "marca_norm" in df_reference_for_freq.columns:
        freq = df_reference_for_freq["marca_norm"].value_counts(dropna=False)
        row["marca_freq"] = row["marca_norm"].map(freq).astype("Int64")

        top_brands = set(df_reference_for_freq["marca_norm"].value_counts().head(config.top_n_brands).index)
        row["marca_topN"] = np.where(row["marca_norm"].isin(top_brands), row["marca_norm"], "OTRAS")
    else:
        row["marca_freq"] = pd.Series([1], dtype="Int64")
        row["marca_topN"] = "OTRAS"

    return row


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera CSV enriched desde el CSV original.")
    p.add_argument("--in", dest="input_path", required=True, help="Ruta del CSV original (ej: Unsupervised_Learning.csv)")
    p.add_argument("--out", dest="output_path", required=True, help="Ruta del CSV enriched de salida")
    p.add_argument("--topn", dest="topn", type=int, default=20, help="Top N marcas a conservar en marca_topN (default: 20)")
    p.add_argument("--encoding", dest="encoding", default="utf-8", help="Encoding del CSV (default: utf-8)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input_path, encoding=args.encoding)

    config = FEConfig(top_n_brands=args.topn)
    df_enriched = build_enriched_features(df, config=config)

    df_enriched.to_csv(args.output_path, index=False, encoding=args.encoding)
    print(f"[OK] Generado: {args.output_path}")
    print(f"     Filas: {df_enriched.shape[0]} | Columnas: {df_enriched.shape[1]}")


if __name__ == "__main__":
    main()
