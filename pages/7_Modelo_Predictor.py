# pages/7_Modelo_Predictor.py

import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# opcional local
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

# Si te da model_not_found, cambia por otro disponible en tu cuenta
LLM_MODEL = "gpt-4o-mini"


# =========================
# PAGE
# =========================
st.set_page_config(page_title="Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Predictor: Segmento de mercado + Precio (con explicación LLM)")
st.caption("Segmentación usa precio ingresado. Predicción de precio NO requiere precio.")


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

    # Lookups
    marca_to_segmento = df.groupby("marca")["segmento_marca"].apply(safe_mode).to_dict()
    marca_to_origen = df.groupby("marca")["origen_marca"].apply(safe_mode).to_dict()
    mm_to_segmento = df.groupby(["marca", "modelo"])["segmento_marca"].apply(safe_mode).to_dict()
    mm_to_origen = df.groupby(["marca", "modelo"])["origen_marca"].apply(safe_mode).to_dict()

    # Participación de mercado por marca (frecuencia relativa en tu dataset final)
    marca_counts = df["marca"].value_counts()
    total = float(len(df))
    marca_to_part = (marca_counts / total).to_dict()

    # dropdowns para UI
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
    """Autocompleta variables necesarias para los modelos (interno, NO UI)."""
    car = dict(base)
    marca = car["marca"]
    modelo = car["modelo"]

    if (marca, modelo) not in lookups["allowed_pairs"]:
        raise ValueError(f"(marca, modelo)=({marca}, {modelo}) no existe en el catálogo del CSV.")

    car["participacion_mercado"] = lookups["marca_to_part"].get(marca, np.nan)
    car["segmento_marca"] = lookups["mm_to_segmento"].get((marca, modelo), lookups["marca_to_segmento"].get(marca, np.nan))
    car["origen_marca"] = lookups["mm_to_origen"].get((marca, modelo), lookups["marca_to_origen"].get(marca, np.nan))
    return car


def predict_segment(car_enriched: dict, models: dict) -> str:
    model = models["cluster_model"]
    le = models["label_encoder"]
    feats = models["cluster_num"] + models["cluster_cat"]

    df_input = pd.DataFrame([car_enriched])[feats]
    pred_enc = model.predict(df_input)[0]
    return le.inverse_transform([pred_enc])[0]


def predict_price(car_enriched: dict, models: dict) -> float:
    model = models["price_model"]
    feats = models["price_num"] + models["price_cat"]

    df_input = pd.DataFrame([car_enriched])[feats]
    return float(model.predict(df_input)[0])


# =========================
# LLM helpers
# =========================
def build_llm_prompt(payload: dict, mode: str) -> str:
    if mode == "segmento":
        intro = (
            "Eres un analista del mercado de autos usados en Costa Rica. "
            "Explica por qué este vehículo cae en el segmento predicho."
        )
    else:
        intro = (
            "Eres un analista del mercado de autos usados en Costa Rica. "
            "Explica por qué este vehículo tiene el precio estimado."
        )

    schema = """
Devuelve SOLO un JSON válido con esta estructura exacta:

{
  "puntos_claros": [
    {
      "feature": "nombre_de_variable (ej: kilometraje)",
      "valor": "valor usado (ej: 85000)",
      "punto": "título corto",
      "descripcion": "1-2 frases claras",
      "impacto": "sube|baja|neutro"
    }
  ],
  "resumen": "1 frase final"
}

Reglas:
- 6 a 8 puntos en "puntos_claros"
- Cada punto DEBE referirse a una feature del JSON de entrada (usa feature+valor)
- Usa SOLO el JSON de entrada (no inventes datos externos)
- NO uses conocimiento general de marcas/modelos (ej: 'BMW es lujo') si no está en el JSON
- No incluyas texto fuera del JSON
"""
    return f"{intro}\n{schema}\n\nJSON de entrada:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"


def extract_json(text: str):
    """Extrae JSON aunque venga con texto extra alrededor."""
    if not text:
        return None
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


def call_llm(prompt: str) -> str:
    """Compat: SDK nuevo (responses) o viejo (chat.completions)."""
    if not OPENAI_API_KEY:
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # SDK nuevo
        if hasattr(client, "responses"):
            resp = client.responses.create(model=LLM_MODEL, input=prompt)
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text

            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        parts.append(t)
            return "\n".join(parts).strip()

        # SDK viejo
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"__ERROR__:{type(e).__name__}:{e}"


def render_llm_explanation(result: dict):
    """Opción 1: render bonito, referido por feature y valor."""
    st.subheader("📌 Explicación (LLM)")

    puntos = result.get("puntos_claros", [])
    resumen = result.get("resumen", "")

    if not isinstance(puntos, list) or len(puntos) == 0:
        st.warning("No se recibieron puntos claros en el formato esperado.")
        return

    def badge(impacto: str) -> str:
        impacto = (impacto or "").strip().lower()
        if impacto == "sube":
            return "🟢 Sube"
        if impacto == "baja":
            return "🔴 Baja"
        return "🟡 Neutro"

    for item in puntos:
        feature = str(item.get("feature", "")).strip()
        valor = str(item.get("valor", "")).strip()
        punto = str(item.get("punto", "")).strip()
        desc = str(item.get("descripcion", "")).strip()
        impacto = badge(item.get("impacto", ""))

        header = f"**{feature}: {valor}**" if (feature or valor) else "**Característica**"
        title = f"✔️ {punto}" if punto else "✔️ Punto"

        st.markdown(f"""
{header}  ·  {impacto}  
**{title}**  
{desc}
""")

    if resumen:
        st.divider()
        st.markdown(f"**🧠 Resumen:** {resumen}")


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
# UI helpers
# =========================
def render_common_inputs(prefix: str):
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
    st.subheader("Segmentación (XGBoost)")
    base = render_common_inputs(prefix="seg")
    precio_crc = st.number_input("Precio (CRC) (obligatorio para segmentación)", min_value=0, value=14500000, step=100000)

    use_llm = st.toggle("Explicar con LLM", value=True, key="seg_llm")
    run = st.button("🔮 Predecir segmento", type="primary", key="seg_run")

    if run:
        try:
            base["precio_crc"] = precio_crc
            car_enriched = enrich_from_brand_model(base, lookups)

            segmento = predict_segment(car_enriched, models)
            st.metric("Segmento", str(segmento))

            if use_llm:
                if not OPENAI_API_KEY:
                    st.warning("OPENAI_API_KEY no está configurada. Configúrala en Secrets/env para usar el LLM.")
                else:
                    payload = {
                        "marca": car_enriched["marca"],
                        "modelo": car_enriched["modelo"],
                        "precio_crc": float(precio_crc),

                        "kilometraje": float(car_enriched["kilometraje"]),
                        "antiguedad": float(car_enriched["antiguedad"]),
                        "cilindrada": float(car_enriched["cilindrada"]),
                        "puertas": int(car_enriched["puertas"]),
                        "pasajeros": int(car_enriched["pasajeros"]),

                        "estilo": car_enriched["estilo"],
                        "combustible": car_enriched["combustible"],
                        "transmision": car_enriched["transmision"],
                        "estado": car_enriched["estado"],
                        "provincia": car_enriched["provincia"],

                        "segmento_predicho": segmento,
                    }

                    prompt = build_llm_prompt(payload, mode="segmento")

                    with st.spinner("Generando explicación..."):
                        raw = call_llm(prompt)

                    if raw.startswith("__ERROR__:"):
                        _, etype, emsg = raw.split(":", 2)
                        st.warning(f"No se pudo generar explicación: {etype}: {emsg}")
                    else:
                        parsed = extract_json(raw)
                        if parsed and isinstance(parsed, dict):
                            render_llm_explanation(parsed)
                        else:
                            st.warning("El LLM no devolvió JSON válido. Mostrando texto crudo:")
                            st.code(raw)

        except Exception as e:
            st.error(f"Error: {e}")


with tab_price:
    st.subheader("Predicción de precio (CatBoost)")
    base = render_common_inputs(prefix="price")

    use_llm = st.toggle("Explicar con LLM", value=True, key="price_llm")
    run = st.button("💰 Predecir precio", type="primary", key="price_run")

    if run:
        try:
            car_enriched = enrich_from_brand_model(base, lookups)
            precio_pred = predict_price(car_enriched, models)

            st.metric("Precio estimado (CRC)", f"₡{precio_pred:,.0f}")

            if use_llm:
                if not OPENAI_API_KEY:
                    st.warning("OPENAI_API_KEY no está configurada. Configúrala en Secrets/env para usar el LLM.")
                else:
                    payload = {
                        "marca": car_enriched["marca"],
                        "modelo": car_enriched["modelo"],

                        "kilometraje": float(car_enriched["kilometraje"]),
                        "antiguedad": float(car_enriched["antiguedad"]),
                        "cilindrada": float(car_enriched["cilindrada"]),
                        "puertas": int(car_enriched["puertas"]),
                        "pasajeros": int(car_enriched["pasajeros"]),

                        "estilo": car_enriched["estilo"],
                        "combustible": car_enriched["combustible"],
                        "transmision": car_enriched["transmision"],
                        "estado": car_enriched["estado"],
                        "provincia": car_enriched["provincia"],

                        "precio_estimado_crc": float(precio_pred),
                    }

                    prompt = build_llm_prompt(payload, mode="precio")

                    with st.spinner("Generando explicación..."):
                        raw = call_llm(prompt)

                    if raw.startswith("__ERROR__:"):
                        _, etype, emsg = raw.split(":", 2)
                        st.warning(f"No se pudo generar explicación: {etype}: {emsg}")
                    else:
                        parsed = extract_json(raw)
                        if parsed and isinstance(parsed, dict):
                            render_llm_explanation(parsed)
                        else:
                            st.warning("El LLM no devolvió JSON válido. Mostrando texto crudo:")
                            st.code(raw)

        except Exception as e:
            st.error(f"Error: {e}")