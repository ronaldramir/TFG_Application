import streamlit as st

def render_conclusiones():

    st.set_page_config(
        page_title="Conclusiones",
        page_icon="📊",
        layout="wide"
    )

    # =============================
    # HEADER HERO
    # =============================
    st.markdown("""
        <h1 style='text-align: center;'>
        🚗 Análisis del Mercado de Vehículos Usados en Costa Rica
        </h1>
        <h4 style='text-align: center; color: gray;'>
        Metodología CRISP-DM · Machine Learning · Febrero 2026
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # =============================
    # MÉTRICAS DESTACADAS
    # =============================
    st.subheader("📌 Resultados Clave")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Segmentos Identificados", "3", "Clustering Jerárquico (Ward)")
    col2.metric("Clasificación", "98.8%", "XGBoost")
    col3.metric("R² Predicción Precio", "0.78", "CatBoost")
    col4.metric("MAE", "₡1,546,000", "Error promedio")

    st.divider()

    # =============================
    # RESUMEN EJECUTIVO
    # =============================
    with st.container(border=True):
        st.subheader("📄 Resumen Ejecutivo")

        st.write("""
        Este proyecto demuestra la viabilidad de aplicar técnicas avanzadas de Machine Learning
        al mercado de vehículos usados en Costa Rica, utilizando la metodología **CRISP-DM**
        como marco estructural.

        Se analizaron **11,555 registros**, integrando:
        - Web scraping automatizado.
        - Segmentación no supervisada.
        - Clasificación supervisada.
        - Modelos de regresión para estimación de precios.
        - Explicabilidad mediante modelos de lenguaje.
        """)

    # =============================
    # CONCLUSIONES TÉCNICAS
    # =============================
    with st.container(border=True):
        st.subheader("🔎 Conclusiones Técnicas")

        st.markdown("""
        - El mercado presenta **estructura latente clara**, validada mediante *Hierarchical Agglomerative Clustering (Ward)*.
        - Los clusters son **separables y estables**, permitiendo automatización con precisión del 98.8%.
        - El modelo CatBoost alcanza un desempeño sólido considerando la variabilidad del mercado.
        - La combinación de modelos supervisados y no supervisados permite una solución integral.
        """)

    # =============================
    # IMPACTO Y VALOR
    # =============================
    with st.container(border=True):
        st.subheader("🚀 Impacto y Aplicabilidad")

        st.markdown("""
        Este sistema puede utilizarse para:

        - Estimación automatizada de precios de mercado.
        - Clasificación instantánea de nuevos vehículos.
        - Identificación de sobrevaloraciones o subvaloraciones.
        - Soporte a decisiones comerciales y financieras.

        La integración de explicabilidad con IA fortalece la confianza y transparencia del sistema.
        """)

    # =============================
    # FOOTER
    # =============================
    st.divider()

    st.markdown("""
        <div style='text-align: center; color: gray;'>
        <strong>Ronald Ramirez</strong><br>
        Proyecto de Análisis de Datos · 2026<br>
        ronaldramir@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    render_conclusiones()