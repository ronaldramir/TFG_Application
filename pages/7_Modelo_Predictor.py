# pages/7_Modelo_Predictor.py

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Opcional local: en Streamlit Cloud no hace daño
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =========================
# CONFIG
# =========================
LOOKUP_CSV = "data/CR_Autos_FinalRows_with_cluster.csv"

CLUSTER_MODEL_PATH = "models/xgboost_cluster_classifier.joblib"
PRICE_MODEL_PATH = "models/catboost_price_regressor_final.joblib"

# Si te da "model_not_found", cambia a "gpt-4o-mini"
LLM_MODEL = "gpt-4o-mini"


# =========================
# PAGE
# =========================
st.set_page_config(page_title="Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Predictor: Segmento de mercado + Precio (con explicación LLM)")
st.caption(
    "Dos módulos: (1) Segmentación (requiere precio) y (2) Predicción de precio (no requiere precio)."
)


# =========================
# API KEY (Secrets / env)
# =========================
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

    allowed_pairs = set(map(tuple, catalog[["marca", "modelo"]].values))

    # Lookups por marca / marca-modelo
    marca_to_segmento = df.groupby("marca")["segmento_marca"].apply(safe_mode).to_dict()
    marca_to_origen = df.groupby("marca")["origen_marca"].apply(safe_mode).to_dict()

    mm_to_segmento = df.groupby(["marca", "modelo"])["segmento_marca"].apply(safe_mode).to_dict()
    mm_to_origen = df.groupby(["marca", "modelo"])["origen_marca"].apply(safe_mode).to_dict()

    # Participación de mercado por marca (frecuencia relativa en tu dataset final)
    marca_counts = df["marca"].value_counts()
    total = float(len(df))
    marca_to_part = (marca_counts / total).to_dict()

    dropdowns = {}
    for col in ["estilo", "combustible", "transmision", "estado", "provincia"]:
        dropdowns[col] = sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

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


def enrich_from_brand_model(base: dict, lookups: dict) -> dict:
    """Autocompleta variables necesarias para los modelos (sin mostrarlas en UI)."""
    car = dict(base)
    marca = car["marca"]
    modelo = car["modelo"]

    if (marca, modelo) not in lookups["allowed_pairs"]:
        raise ValueError(f"(marca, modelo)=({marca}, {modelo}) no existe en el catálogo del CSV.")

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
    conf = float(np.max(proba_vec))

    return pred_label, conf, proba


def predict_price(car_enriched: dict, models: dict):
    model = models["price_model"]
    feats = models["price_num"] + models["price_cat"]

    df_input = pd.DataFrame([car_enriched])[feats]
    return float(model.predict(df_input)[0])


def explain_with_llm(payload: dict, mode: str) -> str:
    """Explicación robusta: evita AttributeError y muestra error real si falla."""
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY no está configurada. Explicación deshabilitada."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        if mode == "segmento":
            instructions = (
                "Eres un analista del mercado de autos usados en Costa Rica. "
                "Explica por qué este vehículo cae en el segmento indicado usando SOLO el JSON."
            )
        else:
            instructions = (
                "Eres un analista del mercado de autos usados en Costa Rica. "
                "Explica por qué este vehículo tendría el precio estimado usando SOLO el JSON."
            )

        prompt = f"""
{instructions}

Reglas:
- Usa SOLO el JSON.
- No inventes datos externos.
- Da 6–8 puntos claros.
- Cierra con un resumen en una sola frase.

JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

        resp = client.responses.create(
            model=LLM_MODEL,
            input=prompt,
        )

        # Ruta 1: algunas versiones ofrecen output_text
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text

        # Ruta 2: fallback genérico
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if t:
                    parts.append(t)

        text = "\n".join(parts).strip()
        return text if text else "⚠️ El LLM respondió, pero no se pudo extraer texto."

    except Exception as e:
        return f"⚠️ No se pudo generar explicación: {type(e).__name__}: {e}"


# =========================
# INIT
# =========================
try:
    models = load_models()
    lookups = load_lookup()
except Exception as e:
    st.error(f"No se pudieron cargar recursos (modelos/CSV). Error: {e}")
    st.stop()

catalog = lookups["catalog"]
dropdowns = lookups["dropdowns"]


# =========================
# UI HELPERS
# =========================
def render_common_inputs(prefix: str):
    """Inputs comunes (marca/modelo + características). prefix para evitar colisiones entre tabs."""
    marcas = sorted(catalog["marca"].unique().tolist())
    marca = st.selectbox("Marca", marcas, key=f"{prefix}_marca")

    modelos_disp = catalog.loc[catalog["marca"] == marca, "modelo"].sort_values().tolist()
    modelo = st.selectbox("Modelo", modelos_disp, key=f"{prefix}_modelo")

    c1, c2 = st.columns(2)
    with c1:
        kilometraje = st.number_input("Kilometraje", min_value=0, value=85000, step=1000, key=f"{prefix}_km")
        antiguedad = st.number_input("Antigüedad (años)", min_value=0, value=6, step=1, key=f"{prefix}_ant")
        cilindrada = st.number_input("Cilindrada (cc)", min_value=0, value=2500, step=100, key=f"{prefix}_cil")
        puertas = st.number_input("Puertas", min_value=2, max_value=7, value=5, step=1, key=f"{prefix}_pue")
    with c2:
        pasajeros = st.number_input("Pasajeros", min_value=1, max_value=12, value=5, step=1, key=f"{prefix}_pas")
        estado_opts = dropdowns.get("estado") or ["Usado", "Nuevo", "Seminuevo"]
        estado = st.selectbox("Estado", estado_opts, key=f"{prefix}_estado")
        prov_opts = dropdowns.get("provincia") or ["San José","Alajuela","Cartago","Heredia","Guanacaste","Puntarenas","Limón"]
        provincia = st.selectbox("Provincia", prov_opts, key=f"{prefix}_prov")

    c3, c4, c5 = st.columns(3)
    with c3:
        estilo_opts = dropdowns.get("estilo") or ["SUV","Sedán","Hatchback","Pick-up"]
        estilo = st.selectbox("Estilo", estilo_opts, key=f"{prefix}_estilo")
    with c4:
        comb_opts = dropdowns.get("combustible") or ["Gasolina","Diésel","Híbrido","Eléctrico"]
        combustible = st.selectbox("Combustible", comb_opts, key=f"{prefix}_comb")
    with c5:
        trans_opts = dropdowns.get("transmision") or ["Automática","Manual"]
        transmision = st.selectbox("Transmisión", trans_opts, key=f"{prefix}_trans")

    return {
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


# =========================
# TABS
# =========================
tab_seg, tab_price = st.tabs(["📌 Segmentación (requiere precio)", "💰 Predicción de precio (sin precio)"])

with tab_seg:
    st.subheader("Segmentación (XGBoost) usando precio ingresado")
    st.write("Este módulo usa el modelo tal como fue entrenado, por eso necesita **precio_crc**.")

    base = render_common_inputs(prefix="seg")
    precio_crc = st.number_input(
        "Precio (CRC) (obligatorio para segmentación)",
        min_value=0, value=14500000, step=100000,
        key="seg_precio"
    )

    use_llm = st.toggle("Explicar con LLM", value=True, key="seg_llm")
    run = st.button("🔮 Predecir segmento", type="primary", key="seg_run")

    if run:
        try:
            base["precio_crc"] = precio_crc
            car_enriched = enrich_from_brand_model(base, lookups)

            segmento, conf, proba = predict_segment(car_enriched, models)

            st.metric("Segmento", str(segmento))

            st.markdown("---")
            st.subheader("Probabilidades (segmento)")
            proba_df = (
                pd.DataFrame([proba]).T.reset_index()
                .rename(columns={"index": "segmento", 0: "probabilidad"})
                .sort_values("probabilidad", ascending=False)
            )
            st.dataframe(proba_df, use_container_width=True)

            if use_llm:
                # Payload “limpio”: no mostramos ni enfatizamos variables autocompletadas
                payload = {
                    "marca": car_enriched["marca"],
                    "modelo": car_enriched["modelo"],
                    "precio_ingresado_crc": precio_crc,
                    "kilometraje": car_enriched["kilometraje"],
                    "antiguedad": car_enriched["antiguedad"],
                    "cilindrada": car_enriched["cilindrada"],
                    "puertas": car_enriched["puertas"],
                    "pasajeros": car_enriched["pasajeros"],
                    "estilo": car_enriched["estilo"],
                    "combustible": car_enriched["combustible"],
                    "transmision": car_enriched["transmision"],
                    "segmento_predicho": segmento,
                    "probabilidades": proba,
                }

                with st.spinner("Generando explicación..."):
                    st.write(explain_with_llm(payload, mode="segmento"))

        except Exception as e:
            st.error(f"Error: {e}")


with tab_price:
    st.subheader("Predicción de precio (CatBoost) SIN ingresar precio")
    st.write("Aquí el usuario no ingresa precio. El modelo lo estima usando las variables del vehículo.")

    base = render_common_inputs(prefix="price")

    use_llm = st.toggle("Explicar con LLM", value=True, key="price_llm")
    run = st.button("💰 Predecir precio", type="primary", key="price_run")

    if run:
        try:
            car_enriched = enrich_from_brand_model(base, lookups)
            precio_pred = predict_price(car_enriched, models)

            st.metric("Precio estimado (CRC)", f"₡{precio_pred:,.0f}")

            if use_llm:
                payload = {
                    "marca": car_enriched["marca"],
                    "modelo": car_enriched["modelo"],
                    "kilometraje": car_enriched["kilometraje"],
                    "antiguedad": car_enriched["antiguedad"],
                    "cilindrada": car_enriched["cilindrada"],
                    "puertas": car_enriched["puertas"],
                    "pasajeros": car_enriched["pasajeros"],
                    "estilo": car_enriched["estilo"],
                    "combustible": car_enriched["combustible"],
                    "transmision": car_enriched["transmision"],
                    "precio_estimado_crc": precio_pred,
                }

                with st.spinner("Generando explicación..."):
                    st.write(explain_with_llm(payload, mode="precio"))

        except Exception as e:
            st.error(f"Error: {e}")