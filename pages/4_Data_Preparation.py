import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Preparación de datos | Feature Engineering",
    page_icon="🧪",
    layout="centered"
)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
with st.container(border=True):
    st.title("🧪 Preparación de datos y variables derivadas")
    st.caption("Transformaciones aplicadas al dataset limpio para generar el CSV enriquecido (enriched)")

st.write("")

# ------------------------------------------------------------
# Objetivo
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🎯 Objetivo de esta etapa")
    st.markdown("""
En esta fase se parte de **`CR_Autos_Cleaned.csv`** (dataset ya limpio) y se generan variables derivadas para:
- **Estandarizar** la marca (evitar duplicados por escritura inconsistente)
- Añadir **contexto de negocio** (origen y segmento de la marca)
- Medir **participación de mercado** por marca como proporción del total

El resultado final se guarda como **`CR_Autos_Cleaned_enriched.csv`**.
""")

st.write("")

# ------------------------------------------------------------
# Inputs / Outputs
# ------------------------------------------------------------
with st.container(border=True):
    st.header("📦 Entrada y salida")
    st.markdown("""
- **Entrada:** `CR_Autos_Cleaned.csv`  
- **Salida:** `CR_Autos_Cleaned_enriched.csv`  

La salida mantiene todas las columnas originales y agrega:
- `marca_norm`
- `origen_marca`
- `segmento_marca`
- `participacion_mercado` (proporción, útil para mostrar como %)
""")

st.write("")

# ------------------------------------------------------------
# Paso 1: Normalización de marca
# ------------------------------------------------------------
with st.container(border=True):
    st.header("1️⃣ Normalización de marca (`marca_norm`)")
    st.markdown("""
Se crea `marca_norm` a partir de `marca` para:
- Quitar espacios
- Pasar a mayúsculas
- Convertir textos basura (`NAN`, `NONE`, vacío) a nulos
- Corregir alias frecuentes (ej. `DONFENG → DONGFENG`, `SSANG → SSANGYONG`)
""")

st.write("")

# ------------------------------------------------------------
# Paso 2: Origen de marca
# ------------------------------------------------------------
with st.container(border=True):
    st.header("2️⃣ Origen de marca (`origen_marca`)")
    st.markdown("""
Se asigna un origen geográfico aproximado por marca usando un diccionario (ejemplos: Japón, Corea, Alemania, USA, Francia, China).
Las marcas no encontradas quedan como **`DESCONOCIDO`**.
""")

st.write("")

# ------------------------------------------------------------
# Paso 3: Segmento de marca
# ------------------------------------------------------------
with st.container(border=True):
    st.header("3️⃣ Segmento de marca (`segmento_marca`)")
    st.markdown("""
Se asigna un segmento de negocio por marca usando un diccionario:
- **PREMIUM**
- **GENERALISTA**
- **COMERCIAL**

Si una marca no está mapeada se marca como **`DESCONOCIDO`**.
""")

st.write("")

# ------------------------------------------------------------
# Paso 4: Participación de mercado por marca
# ------------------------------------------------------------
with st.container(border=True):
    st.header("4️⃣ Participación de mercado (`participacion_mercado`)")
    st.markdown("""
Se calcula como:

**participación = (conteo de anuncios de la marca) / (total de anuncios)**

Esto reemplaza el concepto de “frecuencia absoluta” por una medida comparable y fácil de mostrar como **porcentaje (%)**.
""")

st.write("")

# ------------------------------------------------------------
# Código (del notebook)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧩 Implementación en Python (extracto)")
    st.markdown("Este es el núcleo del código usado en tu notebook `Data_Preparation.ipynb`:")

    with st.expander("Ver código", expanded=False):
        st.code(
            """import pandas as pd
import numpy as np

# 0) Cargar dataset final (el que se quiere enriquecer)
df = pd.read_csv("CR_Autos_Cleaned.csv")

# 1) Normalizar la columna marca
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

# 2) Crear variable derivada: origen de marca
brand_origin_map = {
    "TOYOTA": "JAPON", "HONDA": "JAPON", "NISSAN": "JAPON", "MAZDA": "JAPON",
    "MITSUBISHI": "JAPON", "SUZUKI": "JAPON", "SUBARU": "JAPON", "LEXUS": "JAPON",
    "HYUNDAI": "COREA", "KIA": "COREA", "SSANGYONG": "COREA",
    "BMW": "ALEMANIA", "AUDI": "ALEMANIA", "MERCEDES-BENZ": "ALEMANIA",
    "VOLKSWAGEN": "ALEMANIA", "PORSCHE": "ALEMANIA",
    "FORD": "USA", "CHEVROLET": "USA", "JEEP": "USA", "DODGE": "USA", "TESLA": "USA",
    "PEUGEOT": "FRANCIA", "RENAULT": "FRANCIA", "CITROEN": "FRANCIA",
    "CHERY": "CHINA", "GEELY": "CHINA", "BYD": "CHINA", "MG": "CHINA", "GREAT WALL": "CHINA",
}
df["origen_marca"] = df["marca_norm"].map(brand_origin_map).fillna("DESCONOCIDO")

# 3) Crear variable derivada: segmento de marca
brand_segment_map = {
    "BMW": "PREMIUM", "AUDI": "PREMIUM", "MERCEDES-BENZ": "PREMIUM", "PORSCHE": "PREMIUM",
    "LAND ROVER": "PREMIUM", "LEXUS": "PREMIUM", "ROLLS-ROYCE": "PREMIUM",
    "TOYOTA": "GENERALISTA", "HONDA": "GENERALISTA", "NISSAN": "GENERALISTA",
    "HYUNDAI": "GENERALISTA", "KIA": "GENERALISTA", "MAZDA": "GENERALISTA",
    "SUZUKI": "GENERALISTA", "FORD": "GENERALISTA", "CHEVROLET": "GENERALISTA",
    "ISUZU": "COMERCIAL",
}
df["segmento_marca"] = df["marca_norm"].map(brand_segment_map).fillna("DESCONOCIDO")

# 4) Participación de mercado por marca
marca_counts = df["marca_norm"].value_counts(dropna=False)
total_autos = len(df)
df["participacion_mercado"] = df["marca_norm"].map(marca_counts / total_autos)

# 5) Guardar dataset enriquecido
df.to_csv("CR_Autos_Cleaned_enriched.csv", index=False)
""",
            language="python"
        )

st.write("")

# ------------------------------------------------------------
# Vista rápida (si existen archivos al correr la app)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🔎 Vista rápida del resultado (si el archivo existe)")
    st.caption("Esto solo se muestra si tienes el CSV en el mismo directorio donde corre Streamlit.")

    col1, col2 = st.columns(2)
    with col1:
        input_path = st.text_input("Ruta de entrada", value="CR_Autos_Cleaned.csv")
    with col2:
        output_path = st.text_input("Ruta de salida", value="CR_Autos_Cleaned_enriched.csv")

    if st.button("Cargar y mostrar preview"):
        try:
            df_in = pd.read_csv(input_path)
            st.success(f"Entrada cargada: {df_in.shape[0]:,} filas | {df_in.shape[1]} columnas")
            st.dataframe(df_in.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo cargar la entrada. Detalle: {e}")

        try:
            df_out = pd.read_csv(output_path)
            st.success(f"Salida cargada: {df_out.shape[0]:,} filas | {df_out.shape[1]} columnas")

            # Mostrar participación como %
            if "participacion_mercado" in df_out.columns:
                df_prev = df_out[["marca_norm", "participacion_mercado"]].copy()
                df_prev["participacion_mercado_%"] = (df_prev["participacion_mercado"] * 100).round(3)
                st.dataframe(df_prev.drop(columns=["participacion_mercado"]).head(15), use_container_width=True)
            else:
                st.dataframe(df_out.head(10), use_container_width=True)

        except Exception as e:
            st.warning(f"No se pudo cargar la salida (aún). Detalle: {e}")

st.caption("TFG: Preparación de datos | Variables derivadas y dataset enriquecido")