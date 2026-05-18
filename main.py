import streamlit as st
import json
import math
import pandas as pd
from geopy.geocoders import Nominatim
import utm  # For conversion of UTM coordinates to Lat/Lon
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Page Configuration
st.set_page_config(page_title="Planificador Urbano con IA v.1.3", layout="wide", page_icon="🏗️")

# CSS Styling (Preserving your Glassmorphism aesthetic)
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

# ---------------------------------------------------
# 1. DATABASE & AI ENGINES LOAD
# ---------------------------------------------------
@st.cache_data
def cargar_bases_datos():
    try:
        with open('extraccion_municipios.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"Error": {"Sin Datos": {"Falta JSON": {}}}}

@st.cache_resource
def cargar_motor_busqueda():
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
    return [19.35, -99.65] # Fallback to Toluca area

USOS_SUELO_EDOMEX = cargar_bases_datos()

# ---------------------------------------------------
# 2. SIDEBAR INTERFACE (INPUTS & COORDINATE RESOLUTION)
# ---------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="https://cdn-icons-png.flaticon.com/512/2942/2942159.png" width="100">
    </div>
    """, 
    unsafe_allow_html=True)
st.sidebar.title("Parámetros del Proyecto")

# Level 1: Municipality
municipios_disp = list(USOS_SUELO_EDOMEX.keys())
municipio_sel = st.sidebar.selectbox("📍 Municipio", municipios_disp)

# Level 2: Category
categorias_disp = list(USOS_SUELO_EDOMEX[municipio_sel].keys())
categoria_sel = st.sidebar.selectbox("🏙️ Categoría de Uso", categorias_disp)

# Level 3: Land Key
claves_disp = list(USOS_SUELO_EDOMEX[municipio_sel][categoria_sel].keys())
uso_sel = st.sidebar.selectbox("🏷️ Clave de Uso de Suelo", claves_disp)

area_terreno = st.sidebar.number_input("📏 Área del Terreno (m²)", min_value=1.0, value=300.0, step=50.0)
niveles_proyectados = st.sidebar.number_input("🏢 Niveles a Construir", min_value=1, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Ubicación Específica del Predio")
tipo_coordenada = st.sidebar.radio(
    "Formato de entrada:",
    ["Centroide Municipal", "Geográficas (Lat/Lon)", "UTM (Zona 14N)"]
)

# Initialize map view states
map_lat, map_lon = 19.35, -99.65
zoom_level = 11

if tipo_coordenada == "Centroide Municipal":
    map_lat, map_lon = obtener_centroide_municipio(municipio_sel)
    zoom_level = 11
    
elif tipo_coordenada == "Geográficas (Lat/Lon)":
    centroide = obtener_centroide_municipio(municipio_sel)
    lat_input = st.sidebar.number_input("Latitud", format="%.6f", value=centroide[0])
    lon_input = st.sidebar.number_input("Longitud", format="%.6f", value=centroide[1])
    map_lat, map_lon = lat_input, lon_input
    zoom_level = 17  # Deep zoom to show streets clearly

elif tipo_coordenada == "UTM (Zona 14N)":
    # Default placeholder inputs relative to Edomex region
    utm_x = st.sidebar.number_input("Easting (X)", min_value=100000.0, max_value=900000.0, value=432000.0, step=100.0)
    utm_y = st.sidebar.number_input("Northing (Y)", min_value=1000000.0, max_value=3000000.0, value=2140000.0, step=100.0)
    try:
        # Convert UTM Zone 14 Northern Hemisphere to standard Lat/Lon
        lat_conv, lon_conv = utm.to_latlon(utm_x, utm_y, 14, northern=True)
        map_lat, map_lon = lat_conv, lon_conv
        zoom_level = 17  # Deep zoom to show streets clearly
    except Exception as e:
        st.sidebar.error("⚠️ Coordenadas UTM fuera de rango legal.")

regla = USOS_SUELO_EDOMEX[municipio_sel][categoria_sel][uso_sel]

# ---------------------------------------------------
# 3. MAIN DASHBOARD & PROCESSING
# ---------------------------------------------------
st.title("🏗️ Diseño Urbano: Calculadora con chatbot normativo")

col_calc, col_legal = st.columns([2, 1])

with col_calc:
    st.subheader(f"Ubicación Analizada: {municipio_sel}")
    
    # Generate interactive map view with variable street zoom capabilities
    df_mapa = pd.DataFrame({'lat': [map_lat], 'lon': [map_lon]})
    st.map(df_mapa, zoom=zoom_level, use_container_width=True)

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader(f"📊 Análisis Normativo: {uso_sel} ({categoria_sel})")
    
    t_bruto = regla.get("terreno_bruto_m2", "NP")
    cos_porc = regla.get("porcentaje_max_desplante_cos", "NP")
    cus_mult = regla.get("numero_veces_area_predio_cus", "NP")
    niveles_max = regla.get("niveles_max_construccion", "NP")

    if isinstance(t_bruto, (int, float)) and isinstance(cos_porc, (int, float)) and isinstance(cus_mult, (int, float)):
        
        # VALIDATION 1: Substandard lot size
        if area_terreno < t_bruto:
            st.error(f"❌ **Terreno insuficiente:** Tu área ({area_terreno} m²) es menor al Lote Mínimo exigido ({t_bruto} m²). No es posible desarrollar este predio bajo esta normatividad.")
            
        # VALIDATION 2: Height limits breached
        elif isinstance(niveles_max, (int, float)) and niveles_proyectados > niveles_max:
            st.error(f"❌ **Niveles excedidos:** Tienes proyectados {niveles_proyectados} niveles, pero la norma solo permite un máximo de {niveles_max} niveles para la clave {uso_sel}.")
            
        # PASSED ALL CHECKS
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
            
            # Geometric projection alerts against CUS parameters
            area_proyectada = area_desplante * niveles_proyectados
            if area_proyectada > area_construccion_max:
                st.warning(f"⚠️ **Alerta Geométrica:** Si construyes {niveles_proyectados} niveles ocupando todo el desplante ({area_desplante:,.2f} m² por piso), llegarás a {area_proyectada:,.2f} m², lo cual excede tu CUS permitido. Deberás hacer los pisos más pequeños o dejar áreas libres.")
            
    else:
        st.warning("⚠️ **Atención:** Los parámetros para esta clave están catalogados como 'No Permitidos' (NP) o requieren 'Dictamen Técnico' (DT).")
        st.write(f"- Terreno Bruto: {t_bruto} | - COS: {cos_porc} | - CUS: {cus_mult} | - Niveles Max: {niveles_max}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 4. LEGAL CHATBOT ASSISTANT (RAG)
# ---------------------------------------------------
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
                st.markdown("**Artículos más relevantes:**")
                for i, doc in enumerate(documentos_encontrados):
                    fuente = doc.metadata.get("origen", "Documento Oficial")
                    st.success(f"📄 **{fuente}**\n\n{doc.page_content}")
                    
            st.session_state.mensajes.append({"rol": "assistant", "contenido": "Búsqueda completada."})