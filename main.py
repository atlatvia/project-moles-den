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

# 1. Configuración de página y Estilos CSS (Modo Oscuro Premium)
st.set_page_config(page_title="Planificador Urbano con IA v.1.3", layout="wide", page_icon="🏗️")

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
    </style>
""", unsafe_allow_html=True)

# 2. Inicialización estricta de Estados de Sesión (Evita pérdidas de memoria de variables)
if "area_dibujada" not in st.session_state:
    st.session_state.area_dibujada = 0.0
if "area_terreno_val" not in st.session_state:
    st.session_state.area_terreno_val = 300.0
if "usar_mapa" not in st.session_state:
    st.session_state.usar_mapa = False

# 3. Motores de Carga de Información
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

USOS_SUELO_EDOMEX = cargar_bases_datos()

# 4. Panel Lateral de Parámetros
st.sidebar.title("Parámetros del Proyecto")

municipios_disp = list(USOS_SUELO_EDOMEX.keys())
municipio_sel = st.sidebar.selectbox("📍 Municipio", municipios_disp)

categorias_disp = list(USOS_SUELO_EDOMEX[municipio_sel].keys())
categoria_sel = st.sidebar.selectbox("🏙️ Categoría de Uso", categorias_disp)

claves_disp = list(USOS_SUELO_EDOMEX[municipio_sel][categoria_sel].keys())
uso_sel = st.sidebar.selectbox("🏷️ Clave de Uso de Suelo", claves_disp)

# Sincronizador de área interactiva por Callbacks de Streamlit
if st.session_state.area_dibujada > 0:
    st.sidebar.info(f"📐 Área calculada en mapa: {st.session_state.area_dibujada:,.2f} m²")
    
    def trigger_toggle():
        if st.session_state.chk_usar_mapa:
            st.session_state.area_terreno_val = st.session_state.area_dibujada
            st.session_state.usar_mapa = True
        else:
            st.session_state.usar_mapa = False

    st.sidebar.checkbox("Usar la medida del mapa", key="chk_usar_mapa", value=st.session_state.usar_mapa, on_change=trigger_toggle)

# Input numérico con llave estática persistente (Resuelve el bug de los 300m² fijos)
area_terreno = st.sidebar.number_input("📏 Área del Terreno (m²)", min_value=1.0, step=50.0, key="area_terreno_val")
niveles_proyectados = st.sidebar.number_input("🏢 Niveles a Construir", min_value=1, value=2, step=1)

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

    # Renderizado ÚNICO de Folium (Aquí se eliminó por completo el viejo st.map alterno)
    folium_map = folium.Map(location=[map_lat, map_lon], zoom_start=zoom_level, control_scale=True)
    
    draw_plugin = Draw(
        export=False,
        position='topleft',
        draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True, 'polygon': True},
        edit_options={'remove': True}
    )
    draw_plugin.add_to(folium_map)

    # Captura del render
    map_output = st_folium(folium_map, width="100%", height=400, key="interactive_map_engine")

    # Guard de control de ciclo infinito: Compara valores antes de disparar recargas de render
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

    # Contenedor de Análisis Estadístico Metropolitano
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader(f"📊 Análisis Normativo utilizando: {area_terreno:,.2f} m²")
    
    t_bruto = regla.get("terreno_bruto_m2", "NP")
    cos_porc = regla.get("porcentaje_max_desplante_cos", "NP")
    cus_mult = regla.get("numero_veces_area_predio_cus", "NP")
    niveles_max = regla.get("niveles_max_construccion", "NP")

    if isinstance(t_bruto, (int, float)) and isinstance(cos_porc, (int, float)) and isinstance(cus_mult, (int, float)):
        if area_terreno < t_bruto:
            st.error(f"❌ **Terreno insuficiente:** Tu área ({area_terreno:,.2f} m²) es menor al Lote Mínimo exigido ({t_bruto} m²). No es posible desarrollar este predio bajo esta normatividad.")
        elif isinstance(niveles_max, (int, float)) and niveles_proyectados > niveles_max:
            st.error(f"❌ **Niveles excedidos:** Tienes proyectados {niveles_proyectados} niveles, pero la norma solo permite un máximo de {niveles_max} niveles para la clave {uso_sel}.")
        else:
            total_viviendas = math.floor(area_terreno / t_bruto)
            area_desplante = area_terreno * (cos_porc / 100)
            area_construccion_max = area_terreno * cus_mult

            c1, c2, c3 = st.columns(3)
            c1.metric("Lote Mínimo", f"{t_bruto} m²")
            c2.metric("Viviendas Máximas", f"{total_viviendas} viv.")
            c3.metric("Niveles Máximos", f"{niveles_max}")

            st.write("---")
            st.success("✅ **Proyecto Viable. Parámetros máximos de construcción:**")
            st.write(f"🔹 **Área de Desplante Máxima (COS al {cos_porc}%):** {area_desplante:,.2f} m²")
            st.write(f"🔹 **Área de Construcción Total Máxima (CUS {cus_mult}):** {area_construccion_max:,.2f} m²")
            
            area_proyectada = area_desplante * niveles_proyectados
            if area_proyectada > area_construccion_max:
                st.warning(f"⚠️ **Alerta Geométrica:** Si construyes {niveles_proyectados} niveles ocupando todo el desplante ({area_desplante:,.2f} m² por piso), llegarás a {area_proyectada:,.2f} m², lo cual excede tu CUS permitido. Deberás reducir la huella de los pisos superiores.")
    else:
        st.warning("⚠️ **Atención:** Los parámetros para esta clave están catalogados como 'No Permitidos' (NP) o requieren 'Dictamen Técnico' (DT).")
        st.write(f"- Terreno Bruto: {t_bruto} | - COS: {cos_porc} | - CUS: {cus_mult} | - Niveles: {niveles_max}")

    st.markdown('</div>', unsafe_allow_html=True)

# 6. Asistente IA Normativo RAG (Columna Derecha)
with col_legal:
    st.subheader("🤖 Asistente Normativo")
    st.info("Pregunta al Código Administrativo en lenguaje natural.")

    motor = cargar_motor_busqueda()
    if motor is None:
        st.error("No se pudo cargar el asistente IA. Asegúrate de tener la carpeta 'indice_normativo_completo'.")
    else:
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []

        for msg in st.session_state.mensajes:
            with st.chat_message(msg["rol"]):
                st.markdown(msg["contenido"])

        pregunta = st.chat_input("Ej. ¿Cuál es la restricción para corredores urbanos?")

        if pregunta:
            with st.chat_message("user"):
                st.markdown(pregunta)
            st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

            documentos_encontrados = motor.similarity_search(pregunta, k=3)

            with st.chat_message("assistant"):
                st.markdown("**Artículos más relevantes encontrados:**")
                for doc in documentos_encontrados:
                    fuente = doc.metadata.get("origen", "Libro Quinto - Sección Administrativa")
                    st.success(f"📄 **{fuente}**\n\n{doc.page_content}")
            st.session_state.mensajes.append({"rol": "assistant", "contenido": "Búsqueda completada."})