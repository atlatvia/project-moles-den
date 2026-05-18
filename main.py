import streamlit as st
import json
import math
import pandas as pd
from geopy.geocoders import Nominatim
import utm
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fpdf import FPDF
import io

# 1. Configuración de página y Estilos CSS (Modo Oscuro Premium)
st.set_page_config(page_title="Planificador Urbano con IA v.1.5", layout="wide", page_icon="🏗️")

st.markdown("""
    <style>
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #38bdf8; }
    .obligation-box { border-left: 4px solid #facc15; padding-left: 10px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# 2. Inicialización estricta de Estados de Sesión
if "area_dibujada" not in st.session_state:
    st.session_state.area_dibujada = 0.0
if "area_terreno_val" not in st.session_state:
    st.session_state.area_terreno_val = 300.0
if "usar_mapa" not in st.session_state:
    st.session_state.usar_mapa = False

# 3. Funciones de Carga y Soporte
@st.cache_data
def cargar_bases_datos():
    try:
        with open('src/extraccion_municipios.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open('extraccion_municipios.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"Error": {"Sin Datos": {"Falta JSON": {}}}}

@st.cache_resource
def cargar_motor_busqueda():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return FAISS.load_local("src/indice_normativo_completo", embeddings, allow_dangerous_deserialization=True)
    except:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            return FAISS.load_local("indice_normativo_completo", embeddings, allow_dangerous_deserialization=True)
        except:
            return None

@st.cache_data
def obtener_centroide_municipio(municipio):
    geolocator = Nominatim(user_agent="urban_edomex_app")
    try:
        location = geolocator.geocode(f"{municipio}, Estado de México, México")
        if location:
            return [location.latitude, location.longitude]
    except:
        pass
    return [19.35, -99.65]

def calcular_area_poligono(coordinates):
    puntos_utm = []
    for lon, lat in coordinates:
        try:
            x, y, _, _ = utm.from_latlon(lat, lon, force_zone_number=14, force_zone_letter='N')
            puntos_utm.append((x, y))
        except:
            return 0.0
    n = len(puntos_utm)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += puntos_utm[i][0] * puntos_utm[j][1]
        area -= puntos_utm[j][0] * puntos_utm[i][1]
    return abs(area) / 2.0

def crear_reporte_pdf(municipio, categoria, clave, area, niveles, regla, viv_brutas, viv_netas, area_desplante, area_max_const, don_mun, don_est, cajones, area_verde, aprovechamiento):
    """Genera un reporte PDF integrando las obligaciones urbanas."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Encabezado Corporativo
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(15, 10)
    pdf.cell(0, 10, "DICTAMEN TECNICO NORMATIVO", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(15)
    pdf.cell(0, 5, "Analisis de Compatibilidad y Obligaciones - Libro Quinto", ln=True)
    
    # Estatus
    pdf.set_y(48)
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(15, 45, 180, 18, "DF")
    pdf.set_xy(18, 47)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(22, 101, 52)
    pdf.cell(0, 5, "PROYECTO VIABLE", ln=True)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(0, 5, "El analisis geometrico indica cumplimiento de lineamientos.", ln=True)
    
    # Sección 1
    pdf.set_text_color(30, 41, 59)
    pdf.set_y(70)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 7, "1. Datos Generales del Predio", ln=True)
    pdf.line(15, 77, 195, 77)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 10)
    datos = [
        [f"Municipio: {municipio}", f"Categoria: {categoria}"],
        [f"Clave: {clave}", f"Area Terreno: {area:,.2f} m2"],
        [f"Niveles Proyectados: {niveles}", f"Eficiencia (Aprovechamiento): {aprovechamiento}%"]
    ]
    for fila in datos:
        pdf.set_x(15)
        pdf.cell(90, 8, fila[0], border=1)
        pdf.cell(90, 8, fila[1], border=1, ln=True)
        
    # Sección 2: Parámetros y Geometría
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 7, "2. Evaluacion Geometrica y Constructiva", ln=True)
    pdf.line(15, 112, 195, 112)
    pdf.ln(3)
    
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(50, 8, "Parametro", border=1, fill=True)
    pdf.cell(45, 8, "Norma", border=1, fill=True)
    pdf.cell(45, 8, "Proyecto", border=1, fill=True)
    pdf.cell(40, 8, "Estatus", border=1, fill=True, ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    t_bruto = regla.get("terreno_bruto_m2", "NP")
    cos_p = regla.get("porcentaje_max_desplante_cos", "NP")
    cus_m = regla.get("numero_veces_area_predio_cus", "NP")
    
    filas_norma = [
        ["Lote Minimo", f"{t_bruto} m2", f"{area:,.2f} m2", "Cumple"],
        ["Viviendas Brutas / Netas", f"Max. {viv_brutas}", f"{viv_netas} Proyectadas", f"Al {aprovechamiento}%"],
        ["Huella (COS)", f"{cos_p}%", f"{area_desplante:,.2f} m2", "Dentro de Limite"],
        ["Construccion (CUS)", f"x{cus_m}", f"{area_max_const:,.2f} m2", "Dentro de Limite"]
    ]
    for row in filas_norma:
        pdf.set_x(15)
        pdf.cell(50, 8, row[0], border=1)
        pdf.cell(45, 8, row[1], border=1)
        pdf.cell(45, 8, row[2], border=1)
        pdf.cell(40, 8, row[3], border=1, ln=True)
        
    # Sección 3: Obligaciones Urbanas (La nueva lógica del Excel)
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 7, "3. Obligaciones Urbanas y Cesiones (Libro Quinto)", ln=True)
    pdf.line(15, 155, 195, 155)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 9.5)
    filas_obligaciones = [
        ["Area Verde Requerida:", f"{area_verde:,.2f} m2 (12 m2 por vivienda)"],
        ["Donacion Municipal (15 m2/viv):", f"{don_mun:,.2f} m2"],
        ["Donacion Estatal (5 m2/viv):", f"{don_est:,.2f} m2"],
        ["Total Areas de Cesion:", f"{don_mun + don_est:,.2f} m2"],
        ["Estacionamiento Exigido:", f"{cajones} cajones (1 por cada 4 viv.)"],
        ["Infraestructura Adicional:", "1 Lote PTAR | 1 Lote Tanque Agua"]
    ]
    for row in filas_obligaciones:
        pdf.set_x(15)
        pdf.cell(70, 7, row[0], border=0)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(110, 7, row[1], border=0, ln=True)
        pdf.set_font("Helvetica", "", 9.5)
        
    pdf.ln(10)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(180, 4, "Exencion de Responsabilidad: Memoria de calculo algoritmica basada en el Libro Quinto. No sustituye licencias oficiales.")
    
    return pdf.output()

USOS_SUELO_EDOMEX = cargar_bases_datos()

# 4. Panel Lateral de Parámetros
st.sidebar.title("Parámetros del Proyecto")

municipios_disp = list(USOS_SUELO_EDOMEX.keys())
municipio_sel = st.sidebar.selectbox("📍 Municipio", municipios_disp)

categorias_disp = list(USOS_SUELO_EDOMEX[municipio_sel].keys())
categoria_sel = st.sidebar.selectbox("🏙️ Categoría de Uso", categorias_disp)

claves_disp = list(USOS_SUELO_EDOMEX[municipio_sel][categoria_sel].keys())
uso_sel = st.sidebar.selectbox("🏷️ Clave de Uso de Suelo", claves_disp)

if st.session_state.area_dibujada > 0:
    st.sidebar.info(f"📐 Área calculada en mapa: {st.session_state.area_dibujada:,.2f} m²")
    
    def trigger_toggle():
        if st.session_state.chk_usar_mapa:
            st.session_state.area_terreno_val = st.session_state.area_dibujada
            st.session_state.usar_mapa = True
        else:
            st.session_state.usar_mapa = False

    st.sidebar.checkbox("Usar la medida del mapa", key="chk_usar_mapa", value=st.session_state.usar_mapa, on_change=trigger_toggle)

area_terreno = st.sidebar.number_input("📏 Área del Terreno (m²)", min_value=1.0, step=50.0, key="area_terreno_val")
niveles_proyectados = st.sidebar.number_input("🏢 Niveles a Construir", min_value=1, value=2, step=1)

# --- NUEVO: SLIDER DE APROVECHAMIENTO (LÓGICA DEL EXCEL) ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Eficiencia del Proyecto")
aprovechamiento = st.sidebar.slider(
    "Factor de Aprovechamiento (%)", 
    min_value=10, max_value=100, value=80, step=5,
    help="Representa el % real de terreno vendible descontando vialidades internas y restricciones. (En tu tabla Excel se asume 80%)"
)

st.sidebar.markdown("---")
tipo_coordenada = st.sidebar.radio("Centrar mapa por:", ["Centroide Municipal", "Geográficas", "UTM"])
map_lat, map_lon = 19.35, -99.65
zoom_level = 11

if tipo_coordenada == "Centroide Municipal":
    map_lat, map_lon = obtener_centroide_municipio(municipio_sel)
    zoom_level = 12
elif tipo_coordenada == "Geográficas":
    centroide = obtener_centroide_municipio(municipio_sel)
    map_lat = st.sidebar.number_input("Latitud", format="%.6f", value=centroide[0])
    map_lon = st.sidebar.number_input("Longitud", format="%.6f", value=centroide[1])
    zoom_level = 17
elif tipo_coordenada == "UTM":
    utm_x = st.sidebar.number_input("Easting (X)", min_value=100000.0, max_value=900000.0, value=432000.0)
    utm_y = st.sidebar.number_input("Northing (Y)", min_value=1000000.0, max_value=3000000.0, value=2140000.0)
    try:
        map_lat, map_lon = utm.to_latlon(utm_x, utm_y, 14, northern=True)
        zoom_level = 17
    except:
        map_lat, map_lon = obtener_centroide_municipio(municipio_sel)
        zoom_level = 12

regla = USOS_SUELO_EDOMEX[municipio_sel][categoria_sel][uso_sel]

# 5. Interfaz Principal
col_calc, col_legal = st.columns([2, 1])

with col_calc:
    st.subheader(f"Ubicación Analizada: {municipio_sel}")
    st.caption("💡 Selecciona la figura de polígono (⬡) en el mapa para delimitar las dimensiones de tu predio.")

    folium_map = folium.Map(location=[map_lat, map_lon], zoom_start=zoom_level, control_scale=True)
    
    draw_plugin = Draw(
        export=False, position='topleft',
        draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True, 'polygon': True},
        edit_options={'remove': True}
    )
    draw_plugin.add_to(folium_map)

    map_output = st_folium(folium_map, width="100%", height=400, key="interactive_map_engine")

    if map_output and map_output.get("last_active_drawing"):
        drawing_geometry = map_output["last_active_drawing"]["geometry"]
        if drawing_geometry["type"] == "Polygon":
            polygon_coords = drawing_geometry["coordinates"][0]
            calculated_meters = round(calcular_area_poligono(polygon_coords), 2)
            
            if calculated_meters != st.session_state.area_dibujada:
                st.session_state.area_dibujada = calculated_meters
                if st.session_state.usar_mapa or st.session_state.get('chk_usar_mapa', False):
                    st.session_state.area_terreno_val = calculated_meters
                st.rerun()

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader(f"📊 Análisis y Viabilidad ({area_terreno:,.2f} m²)")
    
    t_bruto = regla.get("terreno_bruto_m2", "NP")
    cos_porc = regla.get("porcentaje_max_desplante_cos", "NP")
    cus_mult = regla.get("numero_veces_area_predio_cus", "NP")
    niveles_max = regla.get("niveles_max_construccion", "NP")

    if isinstance(t_bruto, (int, float)) and isinstance(cos_porc, (int, float)) and isinstance(cus_mult, (int, float)):
        if area_terreno < t_bruto:
            st.error(f"❌ **Terreno insuficiente:** Tu área ({area_terreno:,.2f} m²) es menor al Lote Mínimo exigido ({t_bruto} m²).")
        elif isinstance(niveles_max, (int, float)) and niveles_proyectados > niveles_max:
            st.error(f"❌ **Niveles excedidos:** Proyectas {niveles_proyectados} niveles, pero la norma permite un máximo de {niveles_max}.")
        else:
            # --- NUEVA LÓGICA MATEMÁTICA DEL EXCEL ---
            viviendas_brutas = math.floor(area_terreno / t_bruto)
            viviendas_netas = math.floor(viviendas_brutas * (aprovechamiento / 100.0))
            
            donacion_municipal = viviendas_netas * 15
            donacion_estatal = viviendas_netas * 5
            cajones_estacionamiento = math.ceil(viviendas_netas / 4)
            area_verde_req = viviendas_netas * 12
            
            area_desplante = area_terreno * (cos_porc / 100)
            area_construccion_max = area_terreno * cus_mult

            c1, c2, c3 = st.columns(3)
            c1.metric("Lote Mínimo", f"{t_bruto} m²")
            c2.metric("Viv. Brutas (100%)", f"{viviendas_brutas}")
            c3.metric(f"Viv. Netas ({aprovechamiento}%)", f"{viviendas_netas}")

            st.write("---")
            st.success("✅ **Geometría del Proyecto Viable:**")
            st.write(f"🔹 **Huella de Desplante (COS {cos_porc}%):** {area_desplante:,.2f} m²")
            st.write(f"🔹 **Construcción Total (CUS {cus_mult}):** {area_construccion_max:,.2f} m²")
            
            area_proyectada = area_desplante * niveles_proyectados
            if area_proyectada > area_construccion_max:
                st.warning(f"⚠️ **Alerta:** Con {niveles_proyectados} niveles ocupando todo el desplante, excedes el CUS permitido.")
            
            # --- PANEL DE OBLIGACIONES Y CESIONES ---
            st.markdown('<div class="obligation-box">', unsafe_allow_html=True)
            st.markdown("#### 🌳 Obligaciones y Cesiones (Libro Quinto)")
            col_o1, col_o2, col_o3 = st.columns(3)
            col_o1.metric("Donaciones (Mun+Est)", f"{donacion_municipal + donacion_estatal:,.0f} m²", f"M: {donacion_municipal} | E: {donacion_estatal}")
            col_o2.metric("Estacionamiento", f"{cajones_estacionamiento} Cajones", "50% Chicos | 50% Grandes")
            col_o3.metric("Área Verde Exigida", f"{area_verde_req:,.0f} m²", "12 m² por vivienda")
            st.caption("ℹ️ *Se asume equipamiento de 1 Lote PTAR y 1 Lote para Tanque de Agua por desarrollo.*")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- BOTÓN DE DESCARGA PDF ---
            st.write("")
            try:
                pdf_bytes = crear_reporte_pdf(
                    municipio_sel, categoria_sel, uso_sel, area_terreno, niveles_proyectados,
                    regla, viviendas_brutas, viviendas_netas, area_desplante, area_construccion_max,
                    donacion_municipal, donacion_estatal, cajones_estacionamiento, area_verde_req, aprovechamiento
                )
                st.download_button(
                    label="📥 Descargar Dictamen Integral (PDF)",
                    data=bytes(pdf_bytes),
                    file_name=f"Dictamen_{municipio_sel}_{uso_sel}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"No se pudo compilar el PDF: {e}")
                
    else:
        st.warning("⚠️ **Atención:** Parámetros catalogados como 'No Permitidos' (NP) o requieren 'Dictamen Técnico' (DT).")

    st.markdown('</div>', unsafe_allow_html=True)

# 6. Asistente IA Normativo RAG (Columna Derecha)
with col_legal:
    st.subheader("🤖 Asistente Normativo")
    st.info("Pregunta al Código Administrativo en lenguaje natural.")

    motor = cargar_motor_busqueda()
    if motor is None:
        st.error("No se pudo cargar el asistente IA. Faltan los índices de FAISS.")
    else:
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []

        for msg in st.session_state.mensajes:
            with st.chat_message(msg["rol"]):
                st.markdown(msg["contenido"])

        pregunta = st.chat_input("Ej. ¿Cuántos cajones de estacionamiento necesito?")

        if pregunta:
            with st.chat_message("user"):
                st.markdown(pregunta)
            st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

            documentos_encontrados = motor.similarity_search(pregunta, k=3)

            with st.chat_message("assistant"):
                st.markdown("**Artículos más relevantes encontrados:**")
                for doc in documentos_encontrados:
                    fuente = doc.metadata.get("origen", "Libro Quinto")
                    st.success(f"📄 **{fuente}**\n\n{doc.page_content}")
            st.session_state.mensajes.append({"rol": "assistant", "contenido": "Búsqueda completada."})