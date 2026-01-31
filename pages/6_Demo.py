
import streamlit as st

st.title("🎯 Demo Interactiva")

st.markdown("Aquí irá el formulario para ingresar un vehículo y obtener predicción.")

marca = st.text_input("Marca")
anio = st.number_input("Año", min_value=1990, max_value=2026)
kilometraje = st.number_input("Kilometraje", min_value=0)

if st.button("Predecir"):
    st.success("Aquí se mostrará el precio estimado y el segmento.")
