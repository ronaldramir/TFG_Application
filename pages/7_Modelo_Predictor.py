import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
CLUSTER_MODEL_PATH = "models/xgboost_cluster_classifier.joblib"
PRICE_MODEL_PATH = "models/catboost_price_regressor_final.joblib"  # el final con marca+modelo
LOOKUP_CSV = "data/CR_Autos_FinalRows_with_cluster.csv"

LLM_MODEL = "gpt-4.1-mini"


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Predicción: Segmento + Precio",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Predicción de Segmento de Mercado + Precio (con explicación LLM)")
st.caption(
    "Selecciona un vehículo del catálogo, ingresa sus características, y obtén el segmento, precio estimado y explicación."
)


# =========================
# LOAD ENV (NO subir a git)
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_models():
    cluster_art = joblib.load(CLUSTER_MODEL_PATH)
    price_art = joblib.load(PRICE_MODEL_PATH)

    return {
        "cluster_model": cluster_art["model"],
        "label_encoder": cluster_art["label_encoder"],
        "cluster_num": cluster_art["features_numeric"],
        "cluster_cat": cluster_art["features_categorical"],
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

    # Lookups
    marca_to_segmento = df.groupby("marca")["segmento_marca"].apply(safe_mode).to_dict()
    marca_to_origen = df.groupby("marca")["origen_marca"].apply(safe_mode).to_dict()

    mm_to_segmento = df.groupby(["marca", "modelo"])["segmento_marca"].apply(safe_mode).to_dict()
    mm_to_origen = df.groupby(["marca", "modelo"])["origen_marca"].apply(safe_mode).to_dict()

    # Participación por marca
    marca_counts = df["marca"].value_counts()
    total = float(len(df))
    marca_to_part = (marca_counts / total).to_dict()

    allowed_pairs = set(map(tuple, catalog[["marca", "modelo"]].values))

    return {
        "df": df,
        "catalog": catalog,
        "allowed_pairs": allowed_pairs,
        "marca_to_part": marca_to_part,
        "marca_to_segmento": marca_to_segmento,
        "marca_to_origen": marca_to_origen,
        "mm_to_segmento": mm_to_segmento,
        "mm_to_origen": mm_to_origen,
    }


def enrich_from_brand_model(car: dict, lookups: dict) -> dict:
    car = dict(car)
    marca = car["marca"]
    modelo = car["modelo"]

    if (marca, modelo) not in lookups["allowed_pairs"]:
        raise ValueError(f"(marca, modelo)=({marca}, {modelo}) no existe en el catálogo del CSV.")

    # participación
    car["participacion_mercado"] = lookups["marca_to_part"].get(marca, np.nan)

    # segmento/origen: prioridad marca-modelo, fallback marca
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
    price_pred = float(model.predict(df_input)[0])
    return price_pred


def call_llm_explain(payload: dict, instructions: str) -> str:
    # LLM opcional, no debe romper la app
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY no está configurada. Agrega tu key en .env para habilitar explicación con LLM."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
{instructions}

Reglas:
- Usa SOLO la información del JSON.
- No inventes datos externos.
- Da 6–8 puntos claros.
- Cierra con un resumen en una frase.

JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

        resp = client.responses.create(
            model=LLM_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        return resp.output_text

    except Exception as e:
        return f"⚠️ No se pudo generar explicación con LLM ({type(e).__name__}). La predicción sigue siendo válida."


# =========================
# LOAD RESOURCES
# =========================
models = load_models()
lookups = load_lookup()
catalog = lookups["catalog"]

# =========================
# UI
# =========================
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
        estado = st.selectbox("Estado", ["Usado", "Nuevo", "Seminuevo"], index=0)
        provincia = st.selectbox(
            "Provincia",
            sorted(lookups["df"]["provincia"].dropna().unique().tolist())
            if "provincia" in lookups["df"].columns else
            ["San José","Alajuela","Cartago","Heredia","Guanacaste","Puntarenas","Limón"]
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        estilo = st.selectbox(
            "Estilo",
            sorted(lookups["df"]["estilo"].dropna().unique().tolist())
        )
    with c4:
        combustible = st.selectbox(
            "Combustible",
            sorted(lookups["df"]["combustible"].dropna().unique().tolist())
        )
    with c5:
        transmision = st.selectbox(
            "Transmisión",
            sorted(lookups["df"]["transmision"].dropna().unique().tolist())
        )

    use_llm = st.toggle("Generar explicación con LLM", value=True)
    btn = st.button("🔮 Predecir segmento + precio", type="primary")


with right:
    st.subheader("Resultado")

    if btn:
        # Construir input base
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

            # Predicciones
            segmento, conf, proba = predict_segment(car_enriched, models)
            precio = predict_price(car_enriched, models)

            # KPI display
            k1, k2, k3 = st.columns(3)
            k1.metric("Segmento", segmento)
            k2.metric("Confianza", f"{conf*100:.1f}%")
            k3.metric("Precio estimado", f"₡{precio:,.0f}")

            st.markdown("---")

            st.subheader("Detalle de probabilidades (segmento)")
            proba_df = (
                pd.DataFrame([proba])
                .T.reset_index()
                .rename(columns={"index": "segmento", 0: "probabilidad"})
                .sort_values("probabilidad", ascending=False)
            )
            st.dataframe(proba_df, use_container_width=True)

            st.subheader("Variables autocompletadas desde el CSV")
            st.write(
                {
                    "participacion_mercado": car_enriched.get("participacion_mercado"),
                    "segmento_marca": car_enriched.get("segmento_marca"),
                    "origen_marca": car_enriched.get("origen_marca"),
                }
            )

            st.subheader("Input final utilizado por los modelos")
            st.dataframe(pd.DataFrame([car_enriched]), use_container_width=True)

            # LLM
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

                instructions = (
                    "Eres un analista senior del mercado de autos usados en Costa Rica. "
                    "Explica por qué este vehículo cae en ese segmento y por qué el precio estimado es coherente."
                )

                explanation = call_llm_explain(payload, instructions)
                st.write(explanation)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Completa los campos y presiona **Predecir segmento + precio**.")