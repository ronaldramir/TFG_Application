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
    st.header("💻 Snippet de código")
    st.caption("Ejemplo ilustrativo del patrón usado: búsqueda, paginación, apertura de detalle en nueva pestaña y extracción con control de errores.")
    st.code(
        """

import re, time, random, os, platform, subprocess
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    InvalidSessionIdException,
    WebDriverException,
)

from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions


# ---------------- CONFIG ----------------
URL = "https://crautos.com/autosusados/index.cfm"

YEAR_FROM = "2008"
YEAR_TO   = "2009"
NEWUSED   = "0"  # 0 = Solo usados

# Ritmo (gentil) -> más alto = más estabilidad, menos 500/timeouts
SLEEP_BETWEEN_DETAILS = 1.8
SLEEP_BETWEEN_PAGES   = 3.0
JITTER_DETAILS = (0.3, 1.3)
JITTER_PAGES   = (0.2, 1.0)

# Reintentos detalle
DETAIL_MAX_RETRIES = 7
DETAIL_BASE_SLEEP  = 3.0

# Checkpoint
CHECKPOINT_EVERY_N = 20

# Auto-restart
MAX_DRIVER_RESTARTS = 12

# Cooldown progresivo si hay fallos seguidos
COOLDOWN_STEPS = [0, 10, 25, 50, 90, 180]  # segundos
MAX_CONSECUTIVE_BAD = len(COOLDOWN_STEPS) - 1

# Resume opcional
# CHECKPOINT_TO_RESUME = None
CHECKPOINT_TO_RESUME = "crautos_checkpoint_2008_2009_20251230_193341.csv"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH = f"crautos_checkpoint_{YEAR_FROM}_{YEAR_TO}_{RUN_TS}.csv"
FINAL_PATH      = f"crautos_final_{YEAR_FROM}_{YEAR_TO}_{RUN_TS}.csv"


# ---------------- UTILS ----------------
def _clean_text(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.replace("\xa0", " ").split())

def safe_page_source(driver) -> str:
    try:
        return driver.page_source or ""
    except Exception:
        return ""

def is_500_page(driver) -> bool:
    src = safe_page_source(driver).lower()
    return ("500 - internal server error" in src) or ("server error" in src and "internal server error" in src)

def apply_cooldown(consecutive_bad):
    level = min(consecutive_bad, MAX_CONSECUTIVE_BAD)
    cooldown = COOLDOWN_STEPS[level]
    if cooldown > 0:
        extra = random.uniform(0.0, 2.5)
        print(f"🧊 Cooldown nivel {level}: {cooldown+extra:.1f}s")
        time.sleep(cooldown + extra)

def hard_kill_driver():
    """Mata procesos zombie (principalmente Windows)."""
    try:
        sysname = platform.system().lower()
        if sysname.startswith("win"):
            subprocess.run(["taskkill", "/F", "/IM", "msedgedriver.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "msedge.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "chromedriver.exe", "/T"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "chrome.exe", "/T"], capture_output=True)
        elif sysname == "darwin":
            subprocess.run(["pkill", "-f", "msedgedriver"], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "msedgedriver"], capture_output=True)
    except Exception:
        pass

def checkpoint_save(records, path, last_page, last_idx):
    df = pd.DataFrame(records)
    df["__last_page"] = int(last_page)
    df["__last_idx"]  = int(last_idx)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def load_checkpoint(path):
    df = pd.read_csv(path)
    if "car_id" in df.columns:
        df["car_id"] = df["car_id"].astype(str)

    last_page = 1
    last_idx  = 0
    if "__last_page" in df.columns and df["__last_page"].dropna().shape[0] > 0:
        last_page = int(df["__last_page"].dropna().iloc[-1])
    if "__last_idx" in df.columns and df["__last_idx"].dropna().shape[0] > 0:
        last_idx  = int(df["__last_idx"].dropna().iloc[-1])

    recs = df.drop(columns=[c for c in ["__last_page","__last_idx"] if c in df.columns], errors="ignore") \
             .to_dict(orient="records")
    seen = set(df["car_id"].astype(str).tolist()) if "car_id" in df.columns else set()
    return recs, seen, last_page, last_idx


# ---------------- DRIVER (EDGE) ----------------
def make_driver():
    options = EdgeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")  # opcional

    service = EdgeService()
    driver = webdriver.Edge(service=service, options=options)

    driver.set_page_load_timeout(90)
    driver.set_script_timeout(90)
    return driver


# ---------------- RESULTADOS (cards / paginación) ----------------
def wait_results_ready(driver, wait):
    wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='checkbox' and @name='c']")))

def get_result_cards(driver):
    return driver.find_elements(
        By.XPATH,
        "//div[contains(@class,'card')][.//input[@type='checkbox' and @name='c'] and .//a[contains(@href,'cardetail.cfm?c=')]]"
    )

def get_card_id_and_url(card_we, base_url):
    car_id = card_we.find_element(By.XPATH, ".//input[@type='checkbox' and @name='c']").get_attribute("value")
    a = card_we.find_element(By.XPATH, ".//a[contains(@href,'cardetail.cfm?c=')]")
    href = a.get_attribute("href")
    return str(car_id), urljoin(base_url, href)

def get_active_page_number_safe(driver, retries=10, sleep_s=0.2):
    last_exc = None
    for _ in range(retries):
        try:
            el = driver.find_element(By.CSS_SELECTOR, "ul.pagination li.page-item.active a.page-link")
            txt = _clean_text(el.text)
            return int(re.sub(r"[^\d]", "", txt)) if txt else 1
        except StaleElementReferenceException as e:
            last_exc = e
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    return 1

def has_next_page(driver):
    return len(driver.find_elements(By.CSS_SELECTOR, "ul.pagination li.page-item.page-next a.page-link")) > 0

def click_next_page_stable(driver, wait, timeout=45):
    old_active = driver.find_element(By.CSS_SELECTOR, "ul.pagination li.page-item.active a.page-link")
    nxt = driver.find_element(By.CSS_SELECTOR, "ul.pagination li.page-item.page-next a.page-link")
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", nxt)
    time.sleep(0.25)
    try:
        nxt.click()
    except Exception:
        driver.execute_script("arguments[0].click();", nxt)

    WebDriverWait(driver, timeout).until(EC.staleness_of(old_active))
    wait_results_ready(driver, wait)

def goto_page_via_js(driver, wait, target_page: int, timeout=45):
    current = get_active_page_number_safe(driver)
    if int(target_page) == int(current):
        return
    old_marker = driver.find_element(By.XPATH, "(//input[@type='checkbox' and @name='c'])[1]")
    driver.execute_script("p(arguments[0]);", str(int(target_page)))
    WebDriverWait(driver, timeout).until(EC.staleness_of(old_marker))
    wait_results_ready(driver, wait)

def run_search(driver, wait):
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


# ---------------- DETALLE: abrir/cerrar tab ----------------
def open_detail_in_new_tab_with_retry(driver, wait, detail_url, consecutive_bad):
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
            print(f"⚠️ 500 detalle. retry {attempt}/{DETAIL_MAX_RETRIES}. sleep {sleep_s:.1f}s")
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
            print(f"⚠️ detalle no estable. retry {attempt}/{DETAIL_MAX_RETRIES}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    return False, results_handle

def close_detail_and_back(driver, wait, results_handle):
    driver.close()
    driver.switch_to.window(results_handle)
    wait_results_ready(driver, wait)


# ---------------- EXTRACTORES DETALLE ----------------
def extract_detail_header(driver, wait, timeout=25):
    header = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.carheader"))
    )
    h1s = header.find_elements(By.TAG_NAME, "h1")
    title = _clean_text(h1s[0].text) if len(h1s) >= 1 else ""
    precio_crc_text = _clean_text(h1s[1].text) if len(h1s) >= 2 else ""

    h3s = header.find_elements(By.TAG_NAME, "h3")
    precio_usd_text = _clean_text(h3s[0].text) if len(h3s) >= 1 else ""

    m = re.search(r"(19\d{2}|20\d{2})", title)
    ano = int(m.group(1)) if m else None

    title_wo_year = title.replace(m.group(1), "").strip() if m else title
    parts = title_wo_year.split()
    marca = parts[0] if parts else ""
    modelo = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

    precio_crc = None
    mc = re.search(r"([\d\.,]+)", precio_crc_text)
    if mc:
        precio_crc = int(re.sub(r"[^\d]", "", mc.group(1)))

    precio_usd = None
    mu = re.search(r"([\d\.,]+)", precio_usd_text)
    if mu:
        precio_usd = int(re.sub(r"[^\d]", "", mu.group(1)))

    return {
        "titulo_header": title,
        "marca": marca,
        "modelo": modelo,
        "ano": ano,
        "precio_crc_texto": precio_crc_text,
        "precio_crc": precio_crc,
        "precio_usd_texto": precio_usd_text,
        "precio_usd": precio_usd,
    }

def extract_detail_table(driver, wait, timeout=25):
    table = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-striped.mytext2"))
    )
    rows = table.find_elements(By.CSS_SELECTOR, "tbody > tr")

    kv = {}
    visto_texto = ""
    visto_veces = None
    comentario = ""

    for r in rows:
        tds = r.find_elements(By.TAG_NAME, "td")

        if len(tds) == 2 and tds[0].get_attribute("colspan") is None:
            key = _clean_text(tds[0].text).strip(":").lower()
            val = _clean_text(tds[1].text)
            if key:
                kv[key] = val
            continue

        if len(tds) == 1 and tds[0].get_attribute("colspan") == "2":
            text = _clean_text(tds[0].text)
            if not text:
                continue
            if "visto" in text.lower() and "veces" in text.lower():
                visto_texto = text
                m = re.search(r"(\d[\d,\.]*)", text)
                if m:
                    visto_veces = int(re.sub(r"[^\d]", "", m.group(1)))
            else:
                comentario = text

    def _get(*keys):
        for k in keys:
            if k in kv:
                return kv[k]
        return ""

    return {
        "cilindrada": _get("cilindrada"),
        "estilo": _get("estilo"),
        "pasajeros": _get("# de pasajeros"),
        "combustible": _get("combustible"),
        "transmision": _get("transmisión", "transmision"),
        "estado": _get("estado"),
        "kilometraje": _get("kilometraje"),
        "placa": _get("placa"),
        "color_exterior": _get("color exterior"),
        "color_interior": _get("color interior"),
        "puertas": _get("# de puertas"),
        "impuestos_pagados": _get("ya pagó impuestos", "ya pago impuestos"),
        "precio_negociable": _get("precio negociable"),
        "recibe_vehiculo": _get("se recibe vehículo", "se recibe vehiculo"),
        "provincia": _get("provincia"),
        "fecha_ingreso": _get("fecha de ingreso"),
        "visto_texto": visto_texto,
        "visto_veces": visto_veces,
        "comentario": comentario,
    }


# ---------------- MAIN ----------------
records = []
seen_ids = set()
resume_page = 1
resume_idx  = 0

if CHECKPOINT_TO_RESUME:
    records, seen_ids, resume_page, resume_idx = load_checkpoint(CHECKPOINT_TO_RESUME)
    print(f"✅ Resume: {len(records)} filas | {len(seen_ids)} car_id | last_page={resume_page} last_idx={resume_idx}")
else:
    print("✅ Corrida nueva.")

driver_restarts = 0
consecutive_bad = 0
added_now = 0

driver = make_driver()
wait = WebDriverWait(driver, 45)

last_page = resume_page
last_idx  = resume_idx

try:
    results_url = run_search(driver, wait)
    print("✅ Resultados listos:", results_url)

    if resume_page > 1:
        print(f"↩️ Reanudando: yendo a página {resume_page} ...")
        goto_page_via_js(driver, wait, resume_page)
        time.sleep(1.5)

    while True:
        try:
            active_page = get_active_page_number_safe(driver)
            last_page = active_page
            wait_results_ready(driver, wait)

            cards = get_result_cards(driver)
            total_cards = len(cards)
            print(f"\n=== Página {active_page} === (cards={total_cards}) | rows={len(records)} | bad_streak={consecutive_bad}")

            start_idx = last_idx if active_page == resume_page else 0

            for idx in range(start_idx, total_cards):
                last_idx = idx

                # re-fetch para evitar stale
                cards = get_result_cards(driver)
                card = cards[idx]

                car_id, detail_url = get_card_id_and_url(card, driver.current_url)

                if car_id in seen_ids:
                    continue

                apply_cooldown(consecutive_bad)

                ok, results_handle = open_detail_in_new_tab_with_retry(driver, wait, detail_url, consecutive_bad)
                if not ok:
                    consecutive_bad = min(consecutive_bad + 1, MAX_CONSECUTIVE_BAD)
                    print(f"❌ Saltando car_id={car_id}. bad_streak={consecutive_bad}")
                    continue

                consecutive_bad = 0

                header_data = extract_detail_header(driver, wait)
                table_data  = extract_detail_table(driver, wait)

                rec = {}
                rec.update(header_data)
                rec.update(table_data)
                rec.update({
                    "car_id": car_id,
                    "detail_url": driver.current_url,
                    "pagina": active_page,
                    "posicion_en_pagina": idx + 1,
                })

                records.append(rec)
                seen_ids.add(car_id)
                added_now += 1

                if len(records) % CHECKPOINT_EVERY_N == 0:
                    checkpoint_save(records, CHECKPOINT_PATH, last_page, last_idx)
                    print(f"💾 checkpoint: {CHECKPOINT_PATH} (rows={len(records)}) | last_page={last_page} last_idx={last_idx}")

                close_detail_and_back(driver, wait, results_handle)

                time.sleep(SLEEP_BETWEEN_DETAILS + random.uniform(*JITTER_DETAILS))

            # fin de página
            resume_page = active_page
            last_idx = 0

            if not has_next_page(driver):
                print("✅ Última página. Fin.")
                break

            click_next_page_stable(driver, wait)
            last_idx = 0
            time.sleep(SLEEP_BETWEEN_PAGES + random.uniform(*JITTER_PAGES))

        except (TimeoutException, InvalidSessionIdException, WebDriverException) as e:
            consecutive_bad = min(consecutive_bad + 1, MAX_CONSECUTIVE_BAD)
            print(f"⚠️ Driver/session cayó: {type(e).__name__}. bad_streak={consecutive_bad}. Reiniciando...")

            checkpoint_save(records, CHECKPOINT_PATH, last_page, last_idx)
            print(f"💾 checkpoint antes de reiniciar: {CHECKPOINT_PATH} | last_page={last_page} last_idx={last_idx}")

            try:
                driver.quit()
            except Exception:
                pass

            hard_kill_driver()

            driver_restarts += 1
            if driver_restarts > MAX_DRIVER_RESTARTS:
                raise RuntimeError("Demasiados reinicios de driver; paro por seguridad.")

            apply_cooldown(consecutive_bad)

            driver = make_driver()
            wait = WebDriverWait(driver, 45)

            results_url = run_search(driver, wait)
            print("✅ Re-abierto resultados:", results_url)

            if last_page > 1:
                print(f"↩️ Volviendo a página {last_page} ...")
                goto_page_via_js(driver, wait, last_page)
                time.sleep(1.5)

            continue

    df = pd.DataFrame(records).drop_duplicates(subset=["car_id"], keep="first")
    df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")

    print("\n✅ FIN (EDGE + OPCIÓN A: tabs)")
    print("   + Agregados en esta corrida:", added_now)
    print("   + Total filas:", len(df))
    print("   + car_id únicos:", df["car_id"].nunique())
    print("✅ CSV final:", FINAL_PATH)
    print("✅ Checkpoint:", CHECKPOINT_PATH)

    df.head(10)

except Exception as e:
    print("❌ Error final:", repr(e))
    checkpoint_save(records, CHECKPOINT_PATH, last_page, last_idx)
    print(f"💾 Guardé checkpoint por error: {CHECKPOINT_PATH} | last_page={last_page} last_idx={last_idx}")
    raise


        """,
        language="python"
    )

st.caption("TFG: Analítica del mercado de vehículos usados en Costa Rica | Adquisición de datos")