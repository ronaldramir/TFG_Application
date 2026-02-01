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
    st.caption("Web scraping controlado sobre anuncios públicos de Crautos.com (vehículos usados)")

st.write("")

# ------------------------------------------------------------
# Fuente de datos
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🌐 Fuente de datos")
    st.markdown("""
Los datos provienen del sitio web público **Crautos.com**, específicamente de su sección de vehículos usados.  
La extracción captura los atributos visibles en los anuncios: datos técnicos del vehículo, variables comerciales del anuncio y atributos de ubicación/presentación.
""")

st.write("")

# ------------------------------------------------------------
# Estrategia de recolección
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧩 Estrategia de recolección")
    st.markdown("""
Para mejorar la estabilidad operativa, se implementó scraping **controlado y segmentado por rangos de años** (corridas independientes por año o rangos pequeños).  
Esto permitió:
- Reducir la carga por corrida y mejorar estabilidad  
- Facilitar reintentos y reanudación ante fallos  
- Minimizar pérdidas de progreso por errores temporales del sitio o del navegador  

Adicionalmente, se restringió la búsqueda a usados (`newused = 0`) para mantener consistencia con el objetivo del proyecto.
""")

st.write("")

# ------------------------------------------------------------
# Herramientas y configuración técnica
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🛠️ Herramientas y configuración técnica")
    st.markdown("""
La extracción se desarrolló en **Python** utilizando **Selenium** con **Microsoft Edge WebDriver**.  
El flujo automatizado incluye: carga del formulario, selección de filtros, manejo de paginación y extracción de detalle abriendo el anuncio en una pestaña nueva.
""")

st.write("")

# ------------------------------------------------------------
# Flujo de extracción (alto nivel)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🔁 Flujo de extracción (alto nivel)")
    st.markdown("""
- Ejecutar búsqueda con filtros (rango de años + condición “usados”)  
- Iterar sobre la paginación hasta la última página  
- Por cada card: obtener `car_id` y URL, abrir detalle en nueva pestaña, extraer encabezado y tabla, cerrar y volver  
- Consolidar registros y exportar CSV final deduplicado por `car_id`  
""")

st.write("")

# ------------------------------------------------------------
# Variables capturadas
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧾 Variables capturadas")
    st.markdown("""
Cada anuncio se almacena como una fila con variables del:

**Encabezado:** título, marca, modelo, año (cuando aplica), precio CRC y USD (si está disponible).  
**Tabla:** cilindrada, estilo, combustible, transmisión, estado, kilometraje, colores, puertas, pasajeros, provincia, fecha de ingreso e indicadores comerciales, entre otras.  

Además, se registran variables técnicas para auditoría y control del proceso: `car_id`, `detail_url`, `pagina`, `posicion_en_pagina`.
""")

st.write("")

# ------------------------------------------------------------
# Estabilidad / robustez
# ------------------------------------------------------------
with st.container(border=True):
    st.header("🧯 Estabilidad, errores y contingencias")
    st.markdown("""
Se incorporaron mecanismos de robustez:
- Reintentos con backoff y jitter ante fallos de carga/HTTP 500  
- Control de ritmo con pausas entre detalles y páginas  
- Checkpoints periódicos para reanudar desde el último punto exacto  
- Auto-restart del driver ante fallos de sesión  
- Deduplicación final por `car_id`
""")

st.write("")

# ------------------------------------------------------------
# SNIPPETS DE TU CÓDIGO (REALES)
# ------------------------------------------------------------
with st.container(border=True):
    st.header("💻 Snippets del código")
    st.caption("Extractos reales del script de scraping (configuración, búsqueda y robustez).")

    with st.expander("1) Configuración (rango de años, ritmo, reintentos y checkpoints)", expanded=False):
        st.code(
            """# ---------------- CONFIG ----------------
URL = "https://crautos.com/autosusados/index.cfm"

YEAR_FROM = "2008"
YEAR_TO   = "2009"
NEWUSED   = "0"  # 0 = Solo usados

SLEEP_BETWEEN_DETAILS = 1.8
SLEEP_BETWEEN_PAGES   = 3.0
JITTER_DETAILS = (0.3, 1.3)
JITTER_PAGES   = (0.2, 1.0)

DETAIL_MAX_RETRIES = 7
DETAIL_BASE_SLEEP  = 3.0

CHECKPOINT_EVERY_N = 20
MAX_DRIVER_RESTARTS = 12
""",
            language="python"
        )

    with st.expander("2) Búsqueda con filtros (run_search)", expanded=False):
        st.code(
            """def run_search(driver, wait):
    driver.get(URL)
    form = wait.until(EC.presence_of_element_located((By.ID, "searchform")))

    Select(form.find_element(By.NAME, "yearfrom")).select_by_value(YEAR_FROM)
    Select(form.find_element(By.NAME, "yearto")).select_by_value(YEAR_TO)
    Select(form.find_element(By.NAME, "newused")).select_by_value(NEWUSED)

    btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#searchform button[type='submit']")))
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    time.sleep(0.45)

    try:
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#searchform button[type='submit']")))
        btn.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", btn)
    except Exception:
        driver.execute_script("document.getElementById('searchform').submit();")

    wait_results_ready(driver, wait)
    return driver.current_url
""",
            language="python"
        )

    with st.expander("3) Robustez: abrir detalle en nueva pestaña con retry (HTTP 500 / fallos)", expanded=False):
        st.code(
            """def open_detail_in_new_tab_with_retry(driver, wait, detail_url, consecutive_bad):
    results_handle = driver.current_window_handle
    base_sleep = DETAIL_BASE_SLEEP + consecutive_bad * 1.0

    for attempt in range(1, DETAIL_MAX_RETRIES + 1):
        driver.execute_script("window.open(arguments[0], '_blank');", detail_url)
        driver.switch_to.window(driver.window_handles[-1])

        time.sleep(1.0)

        if is_500_page(driver):
            driver.close()
            driver.switch_to.window(results_handle)
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0.2, 1.6)
            time.sleep(sleep_s)
            continue

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.carheader")))
            return True, results_handle
        except Exception:
            try:
                driver.close()
            except Exception:
                pass
            driver.switch_to.window(results_handle)

            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0.2, 1.6)
            time.sleep(sleep_s)

    return False, results_handle
""",
            language="python"
        )

st.caption("TFG: Adquisición de datos | Web scraping controlado")