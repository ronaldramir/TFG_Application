# pages/7_Modelo_Predictor.py
# ============================================================
# PREDICTOR FINAL (Streamlit)
# - Selección MARCA + MODELO desde catálogo (CSV)
# - Autocompleta: participacion_mercado, segmento_marca, origen_marca
# - Predice: Segmento (XGBoost) + Precio (CatBoost)
# - Explicación con LLM (opcional) usando OPENAI_API_KEY (Secrets / env)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Opcional (local). En Streamlit Cloud no molesta aunque no exista .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =========================
# CONFIG (RUTAS)
# =========================
LOOKUP_CSV = "data/CR_Autos_FinalRows_with_cluster.csv"

CLUSTER_MODEL_PATH = "models/xgboost_cluster_classifier.joblib"
PRICE_MODEL_PATH = "models/catboost_price_regressor_final.joblib"  # el final con marca+modelo

LLM_MODEL = "gpt-4.1-mini"  # modelo para explicación
DEFAULT_USE_LLM = True


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Predictor: Segmento + Precio",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Predictor: Segmento de mercado + Precio (con explicación LLM)")
st.caption(
    "Selecciona marca/modelo del catálogo, ingresa características del vehículo y obtén: "
    "segmento (con confianza), precio estimado y explicación opcional."
)


# =========================
# API KEY (Secrets / env)
# =========================
# Streamlit Cloud: Secrets suelen inyectarse como variables de entorno.
# También soportamos st.secrets por si tu despliegue no las inyecta.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        OPENAI_API_KEY = None


# =========================
# LOAD RESOURCES
# =========================
@st.cache_resource
def load_models():
    cluster_art = joblib.load(CLUSTER_MODEL_PATH)
    price_art = joblib.load(PRICE_MODEL_PATH)

    return {
        # Segmento
        "cluster_model": cluster_art["model"],
        "label_encoder": cluster_art["label_encoder"],
        "cluster_num": cluster_art["features_numeric"],
        "cluster_cat": cluster_art["features_categorical"],
        # Precio
        "price_model": price_art["model"],
        "price_num": price_art["features_numeric"],
        "price_cat": price_art["features_categorical"],
    }


@st.cache_data
def load_lookup():
    df = pd.read_csv(LOOKUP_CSV)

    def safe_mode(s: pd.Series):
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else np.nan

    catalog = (
        df[["marca", "modelo"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["marca", "modelo"])
        .reset_index(drop=True)
    )

    # Lookups por marca
    marca_to_segmento = df.groupby("marca")["segmento_marca"].apply(safe_mode).to_dict()
    marca_to_origen = df.groupby("marca")["origen_marca"].apply(safe_mode).to_dict()

    # Lookups por marca-modelo
    mm_to_segmento = df.groupby(["marca", "modelo"])["segmento_marca"].apply(safe_mode).to_dict()
    mm_to_origen = df.groupby(["marca", "modelo"])["origen_marca"].apply(safe_mode).to_dict()

    # Participación de mercado por marca (frecuencia relativa en el CSV final)
    marca_counts = df["marca"].value_counts()
    total = float(len(df))
    marca_to_part = (marca_counts / total).to_dict()

    allowed_pairs = set(map(tuple, catalog[["marca", "modelo"]].values))

    # Para dropdowns (valores válidos)
    dropdowns = {}
    for col in ["estilo", "combustible", "transmision", "estado", "provincia"]:
        if col in df.columns:
            dropdowns[col] = sorted(df[col].dropna().unique().tolist())
        else:
            dropdowns[col] = []

    return {
        "df": df,
        "catalog": catalog,
        "allowed_pairs": allowed_pairs,
        "marca_to_part": marca_to_part,
        "marca_to_segmento": marca_to_segmento,
        "marca_to_origen": marca_to_origen,
        "mm_to_segmento": mm_to_segmento,
        "mm_to_origen": mm_to_origen,
        "dropdowns": dropdowns,
    }


def enrich_from_brand_model(car: dict, lookups: dict) -> dict:
    car = dict(car)
    marca = car["marca"]
    modelo = car["modelo"]

    if (marca, modelo) not in lookups["allowed_pairs"]:
        raise ValueError(
            f"(marca, modelo)=({marca}, {modelo}) no existe en el catálogo del CSV."
        )

    car["participacion_mercado"] = lookups["marca_to_part"].get(marca, np.nan)

    car["segmento_marca"] = lookups["mm_to_segmento"].get(
        (marca, modelo),
        lookups["marca_to_segmento"].get(marca, np.nan)
    )
    car["origen_marca"] = lookups["mm_to_origen"].get(
        (marca, modelo),
        lookups["marca_to_origen"].get(marca, np.nan)
    )
    return car


def predict_segment(car_enriched: dict, models: dict):
    model = models["cluster_model"]
    le = models["label_encoder"]
    feats = models["cluster_num"] + models["cluster_cat"]

    df_input = pd.DataFrame([car_enriched])[feats]

    pred_enc = model.predict(df_input)[0]
    pred_label = le.inverse_transform([pred_enc])[0]

    proba_vec = model.predict_proba(df_input)[0]
    proba = dict(zip(le.classes_, proba_vec))
    confidence = float(np.max(proba_vec))

    return pred_label, confidence, proba


def predict_price(car_enriched: dict, models: dict):
    model = models["price_model"]
    feats = models["price_num"] + models["price_cat"]

    df_input = pd.DataFrame([car_enriched])[feats]
    return float(model.predict(df_input)[0])


def call_llm_explain(payload: dict) -> str:
    if not OPENAI_API_KEY:
        return (
            "⚠️ OPENAI_API_KEY no está configurada en Secrets/entorno. "
            "La predicción funciona, pero la explicación LLM está deshabilitada."
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        instructions = (
            "Eres un analista senior del mercado de autos usados en Costa Rica. "
            "Explica por qué este vehículo cae en ese segmento y por qué el precio "
            "estimado es coherente."
        )

        prompt = f"""
{instructions}

Reglas:
- Usa SOLO la información del JSON.
- No inventes datos externos (no precios del mercado, no comparaciones con anuncios).
- Da 6–8 puntos claros.
- Cierra con un resumen en una sola frase.

JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

        resp = client.responses.create(
            model=LLM_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        return resp.output_text

    except Exception as e:
        return (
            f"⚠️ No se pudo generar explicación con LLM ({type(e).__name__}). "
            "La predicción sigue siendo válida."
        )


# =========================
# MAIN
# =========================
try:
    models = load_models()
    lookups = load_lookup()
except Exception as e:
    st.error(f"No se pudieron cargar recursos (modelos/CSV). Error: {e}")
    st.stop()

catalog = lookups["catalog"]
dropdowns = lookups["dropdowns"]

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) Selección del vehículo (catálogo)")

    marcas = sorted(catalog["marca"].unique().tolist())
    marca = st.selectbox("Marca", marcas)

    modelos_disponibles = (
        catalog.loc[catalog["marca"] == marca, "modelo"]
        .sort_values()
        .tolist()
    )
    modelo = st.selectbox("Modelo", modelos_disponibles)

    st.subheader("2) Características del vehículo")

    c1, c2 = st.columns(2)

    with c1:
        kilometraje = st.number_input("Kilometraje", min_value=0, value=85000, step=1000)
        antiguedad = st.number_input("Antigüedad (años)", min_value=0, value=6, step=1)
        cilindrada = st.number_input("Cilindrada (cc)", min_value=0, value=2500, step=100)
        puertas = st.number_input("Puertas", min_value=2, max_value=7, value=5, step=1)

    with c2:
        pasajeros = st.number_input("Pasajeros", min_value=1, max_value=12, value=5, step=1)

        estado_options = dropdowns.get("estado") or ["Usado", "Nuevo", "Seminuevo"]
        estado = st.selectbox("Estado", estado_options, index=0)

        provincia_options = dropdowns.get("provincia") or [
            "San José", "Alajuela", "Cartago", "Heredia", "Guanacaste", "Puntarenas", "Limón"
        ]
        provincia = st.selectbox("Provincia", provincia_options)

    c3, c4, c5 = st.columns(3)

    with c3:
        estilo_options = dropdowns.get("estilo") or ["SUV", "Sedán", "Hatchback", "Pick-up"]
        estilo = st.selectbox("Estilo", estilo_options)

    with c4:
        combustible_options = dropdowns.get("combustible") or ["Gasolina", "Diésel", "Híbrido", "Eléctrico"]
        combustible = st.selectbox("Combustible", combustible_options)

    with c5:
        transmision_options = dropdowns.get("transmision") or ["Automática", "Manual"]
        transmision = st.selectbox("Transmisión", transmision_options)

    use_llm = st.toggle("Generar explicación con LLM", value=DEFAULT_USE_LLM)
    btn = st.button("🔮 Predecir segmento + precio", type="primary")


with right:
    st.subheader("Resultado")

    if btn:
        car_input = {
            "marca": marca,
            "modelo": modelo,

            "kilometraje": kilometraje,
            "antiguedad": antiguedad,
            "cilindrada": cilindrada,
            "puertas": puertas,
            "pasajeros": pasajeros,

            "estilo": estilo,
            "combustible": combustible,
            "transmision": transmision,
            "estado": estado,
            "provincia": provincia,
        }

        try:
            car_enriched = enrich_from_brand_model(car_input, lookups)

            segmento, conf, proba = predict_segment(car_enriched, models)
            precio = predict_price(car_enriched, models)

            k1, k2, k3 = st.columns(3)
            k1.metric("Segmento", str(segmento))
            k2.metric("Confianza", f"{conf*100:.1f}%")
            k3.metric("Precio estimado", f"₡{precio:,.0f}")

            st.markdown("---")

            st.subheader("Probabilidades (segmento)")
            proba_df = (
                pd.DataFrame([proba])
                .T.reset_index()
                .rename(columns={"index": "segmento", 0: "probabilidad"})
                .sort_values("probabilidad", ascending=False)
            )
            st.dataframe(proba_df, use_container_width=True)

            st.subheader("Variables autocompletadas (desde CSV)")
            st.write({
                "participacion_mercado": float(car_enriched.get("participacion_mercado")) if pd.notna(car_enriched.get("participacion_mercado")) else None,
                "segmento_marca": car_enriched.get("segmento_marca"),
                "origen_marca": car_enriched.get("origen_marca"),
            })

            st.subheader("Input final usado por los modelos")
            st.dataframe(pd.DataFrame([car_enriched]), use_container_width=True)

            if use_llm:
                st.markdown("---")
                st.subheader("Explicación con LLM")

                payload = {
                    "caracteristicas": car_enriched,
                    "segmento_predicho": segmento,
                    "confianza_segmento": conf,
                    "probabilidades_segmento": proba,
                    "precio_predicho_crc": precio,
                }

                with st.spinner("Generando explicación..."):
                    explanation = call_llm_explain(payload)

                st.write(explanation)

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Completa los campos y presiona **Predecir segmento + precio**.")