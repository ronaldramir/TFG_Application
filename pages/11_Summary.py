import streamlit as st

def render_conclusiones():
    st.set_page_config(page_title="Conclusiones", page_icon="🧾", layout="wide")

    with st.container(border=True):
        st.title("🧾 Conclusiones")
        st.caption("Análisis del Mercado de Vehículos Usados en Costa Rica • Metodología CRISP-DM • Febrero 2026")

    # =========================
    # Resumen Ejecutivo
    # =========================
    with st.container(border=True):
        st.header("Resumen Ejecutivo")
        st.write(
            """
            Este documento presenta un análisis integral del mercado de vehículos usados en Costa Rica,
            desarrollado mediante la metodología **CRISP-DM**. El proyecto abarca desde la extracción
            automatizada de datos hasta la construcción de modelos predictivos para la **segmentación de mercado**
            y **estimación de precios**. El análisis comprende **11,555 registros** extraídos del portal *crautos.com*.
            """
        )

    # =========================
    # Hallazgos principales (métricas)
    # =========================
    with st.container(border=True):
        st.header("Hallazgos Principales")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Segmentos (clustering Ward)", "3")
        c2.metric("Clasificación (XGBoost)", "98.8%")
        c3.metric("Predicción de precios (CatBoost)", "R² = 0.78")
        c4.metric("Error (MAE)", "₡1,546,000")

        st.markdown(
            """
            - **Segmentación de Mercado:** Se identificaron **3 segmentos** principales mediante clustering jerárquico (Ward).
            - **Modelo de Clasificación:** Precisión de **98.8%** con **XGBoost** para asignación automática de segmentos.
            - **Predicción de Precios:** **CatBoost** con **R² = 0.78** y **MAE ≈ ₡1,546,000 CRC**.
            - **Explicabilidad con IA:** Integración de **GPT-4** para explicaciones interpretables.
            """
        )

    # =========================
    # Conclusiones
    # =========================
    with st.container(border=True):
        st.header("Conclusiones")

        st.markdown(
            """
            Este proyecto demuestra la viabilidad y efectividad de aplicar la metodología **CRISP-DM**
            al mercado de vehículos usados en Costa Rica.

            Los modelos desarrollados superaron los criterios de éxito establecidos, logrando:

            - Alta precisión en la **segmentación de mercado** (clustering jerárquico con Ward).
            - Un clasificador robusto (XGBoost) para **asignación automática de segmentos**.
            - Un modelo de regresión (CatBoost) con desempeño sólido para **estimación de precios**.

            La integración de modelos de lenguaje para explicabilidad representa un avance hacia sistemas de IA
            más **transparentes** y **confiables**, acercando el análisis técnico a usuarios no especialistas.
            """
        )

    # =========================
    # Autor / contacto
    # =========================
    with st.container(border=True):
        st.subheader("Autor")
        st.write("**Ronald Ramirez**")
        st.write("Contacto: ronaldramir@gmail.com")


# Si lo ejecutas como página individual:
if __name__ == "__main__":
    render_conclusiones()