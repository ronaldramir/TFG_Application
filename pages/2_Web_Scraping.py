import streamlit as st

st.set_page_config(
    page_title="Web Scraping | Adquisición de datos",
    page_icon="🕷️",
    layout="centered"
)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
with st.container(border=True):
    st.title("🕷️ Origen y adquisición de los datos")
    st.caption("Web scraping controlado y segmentado sobre anuncios públicos de Crautos.com")
    st.markdown("**Objetivo:** recolectar atributos visibles de anuncios de vehículos usados para construir el dataset del proyecto.")

st.write("")

# ------------------------------------------------------------
# Fuente de datos
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🌐 Fuente de datos")
    st.markdown("""
Los datos provienen del sitio web público **Crautos.com**, específicamente de su sección de vehículos usados.  
La extracción se limita a información visible en los anuncios publicados, incluyendo:
- Atributos técnicos del vehículo  
- Variables comerciales del anuncio  
- Atributos de ubicación y presentación  
""")

st.write("")

# ------------------------------------------------------------
# Estrategia de recolección
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧩 Estrategia de recolección")
    st.markdown("""
Debido al volumen de anuncios y a la necesidad de estabilidad operativa durante la extracción, se implementó una estrategia de scraping **controlado y segmentado por rango de años** (por ejemplo, `YEAR_FROM` a `YEAR_TO`), ejecutando corridas independientes por segmentos (año por año o rangos pequeños).  
Esto permitió:
- Reducir la carga por corrida y mejorar la estabilidad del proceso  
- Facilitar reintentos y reanudación en caso de fallos  
- Minimizar pérdidas de progreso ante errores temporales del sitio o del navegador  

Adicionalmente, se restringió la búsqueda a vehículos usados (`newused = 0`) para asegurar consistencia con el objetivo del proyecto.
""")

st.write("")

# ------------------------------------------------------------
# Herramientas y configuración técnica
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🛠️ Herramientas y configuración técnica")
    st.markdown("""
La extracción se desarrolló en **Python** utilizando **Selenium** con **Microsoft Edge WebDriver**.  
El flujo de navegación automatizada consideró:
- Carga del formulario de búsqueda  
- Selección de filtros (año desde/hasta; condición “usados”)  
- Manejo de resultados paginados  
- Apertura del detalle del anuncio en una pestaña nueva para extraer variables  

Se incorporaron configuraciones orientadas a robustez: **timeouts**, **esperas explícitas** y control de interacción con elementos.
""")

st.write("")

# ------------------------------------------------------------
# Flujo de extracción (alto nivel)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🔁 Flujo de extracción (alto nivel)")
    st.markdown("""
El pipeline de adquisición siguió este flujo:

1. Ejecutar búsqueda con filtros definidos (rango de años y condición “usados”).  
2. Iterar sobre la paginación hasta la última página.  
3. En cada página, identificar los resultados (“cards”) y para cada vehículo:  
   - Obtener `car_id` y `detail_url`  
   - Abrir detalle en una pestaña nueva  
   - Extraer variables del encabezado y de la tabla principal  
   - Cerrar pestaña y regresar a resultados  
4. Consolidar registros y exportar un CSV final deduplicado por `car_id`.

Este enfoque separa la extracción de listado (localizar vehículos/URL) y la extracción de detalle (capturar variables completas por anuncio).
""")

st.write("")

# ------------------------------------------------------------
# Variables capturadas
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧾 Variables capturadas y estructura del registro")
    st.markdown("""
Cada anuncio se almacenó como una observación (fila) con variables provenientes de dos fuentes:

**Encabezado del anuncio**
- título, marca, modelo  
- año (parseado desde el título cuando aplica)  
- precio en colones y precio en dólares (cuando está disponible)

**Tabla de características**
- cilindrada, estilo, combustible, transmisión, estado  
- kilometraje, colores, puertas, pasajeros, provincia  
- fecha de ingreso e indicadores comerciales (negociable, impuestos pagados, recibe vehículo), entre otras

Adicionalmente, se registraron variables técnicas para trazabilidad:
- `car_id` (identificador único)  
- `detail_url` (URL visitada)  
- `pagina` y `posicion_en_pagina` (auditoría/depuración)  
""")

st.write("")

# ------------------------------------------------------------
# Estabilidad y contingencias
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧯 Control de estabilidad, errores y contingencias")
    st.markdown("""
Se incorporaron mecanismos explícitos de robustez:

- **Errores HTTP 500 y fallos de carga:** reintentos con backoff exponencial y jitter; omisión del anuncio tras múltiples intentos fallidos.  
- **Control de ritmo:** pausas entre detalles y páginas (`SLEEP_BETWEEN_DETAILS`, `SLEEP_BETWEEN_PAGES`) con aleatoriedad controlada.  
- **Checkpoints y reanudación:** guardado periódico del progreso (`__last_page`, `__last_idx`, `seen_ids`) y de los registros extraídos.  
- **Auto-restart del driver:** ante fallos de sesión, guardado de checkpoint, reinicio del WebDriver y retorno al último punto procesado.  
- **Deduplicación:** salida final deduplicada por `car_id`.  
""")
    st.warning("Nota: Algunos anuncios pueden omitirse si presentan fallos persistentes de carga o error del sitio.")

st.write("")

# ------------------------------------------------------------
# Salidas del proceso
# ------------------------------------------------------------
with st.container(border=True):
    st.header("📦 Salidas del proceso")
    st.markdown("""
El pipeline generó:

- **Checkpoints (intermedios):** recuperación de progreso y auditoría de extracción  
- **Archivo final por corrida:** anuncios extraídos para el rango de años indicado, deduplicado por `car_id`  

Posteriormente, los archivos finales por rango/año se consolidaron en un único dataset maestro (descrito en la fase de preparación o en un anexo técnico).
""")

st.write("")

# ------------------------------------------------------------
# Consideraciones y limitaciones
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧩 Consideraciones y limitaciones")
    st.markdown("""
- Los datos reflejan el **precio publicado**, no el precio real de transacción.  
- La calidad de ciertos campos depende del ingreso manual del anunciante.  
- Algunos anuncios pueden omitirse por fallos persistentes de carga o errores del sitio.  
- El dataset representa el mercado durante la ventana de extracción; el mercado es dinámico y cambia con el tiempo.  
""")

st.write("")

# ------------------------------------------------------------
# Snippet de código (ejemplo)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("💻 Snippet de código (referencial)")
    st.caption("Ejemplo ilustrativo del patrón usado: búsqueda, paginación, apertura de detalle en nueva pestaña y extracción con control de errores.")
    st.code(
        """
# (Ejemplo referencial) Flujo típico del scraping con Selenium

for year_from, year_to in year_ranges:
    apply_filters(driver, year_from=year_from, year_to=year_to, newused=0)
    run_search(driver)

    while not last_page:
        cards = get_result_cards(driver)

        for idx, card in enumerate(cards):
            car_id, detail_url = parse_card(card)

            if car_id in seen_ids:
                continue

            try:
                open_in_new_tab(driver, detail_url)
                record = extract_detail_page(driver, car_id=car_id, page=page, pos=idx)
                results.append(record)
                seen_ids.add(car_id)

            except Exception:
                # reintentos, backoff, skip controlado, logging
                handle_detail_error(car_id, detail_url)

            finally:
                close_tab_and_return(driver)

        go_next_page(driver)
        maybe_checkpoint(results, seen_ids, last_page=page, last_idx=idx)
        """,
        language="python"
    )

st.caption("TFG: Analítica del mercado de vehículos usados en Costa Rica | Adquisición de datos")