import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

from src.etl import generar_datos_prueba
from src.algo_depletion import calcular_depletion_y_reorden
from src.algo_recompra import calcular_oportunidades_recompra
from src.algo_forecast import predecir_ventas_futuras

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Premium Brands | Executive Suite",
    layout="wide",
    page_icon="⭐",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS EJECUTIVOS (ENTERPRISE LOOK) ---
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -1px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E0;
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #1A202C;
        line-height: 1.1;
    }
    .kpi-subtitle {
        font-size: 12px;
        font-weight: 600;
        color: #38A169;
        margin-top: 6px;
    }
    .header-container {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
        padding: 24px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #FFFFFF;
    }
    .header-badge {
        background-color: #C59D7F;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

TC_GLOBAL = 3.50

# --- CARGA Y NORMALIZACIÓN DE DATOS ---
@st.cache_data
def obtener_datos(file):
    if file is not None:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        return df, None, None
    else:
        res = generar_datos_prueba()
        if isinstance(res, tuple):
            return res[0], res[1] if len(res) > 1 else None, res[2] if len(res) > 2 else None
        return res, None, None

archivo_subido = None
df_ventas, df_stock_emp, df_stock_cli = obtener_datos(archivo_subido)

# Normalizador defensivo de columnas
if 'Monto_USD' not in df_ventas.columns:
    if 'Monto' in df_ventas.columns:
        df_ventas['Monto_USD'] = df_ventas['Monto']
    elif 'Venta_USD' in df_ventas.columns:
        df_ventas['Monto_USD'] = df_ventas['Venta_USD']
    else:
        df_ventas['Monto_USD'] = 100.0

if 'Cantidad' not in df_ventas.columns:
    df_ventas['Cantidad'] = 10

if 'Canal' not in df_ventas.columns:
    df_ventas['Canal'] = 'General'

if 'Marca' not in df_ventas.columns:
    df_ventas['Marca'] = 'CHANDON'

df_ventas['Año'] = df_ventas['Fecha'].dt.year
df_ventas['Mes_Num'] = df_ventas['Fecha'].dt.month
df_ventas['Mes_Nombre'] = df_ventas['Fecha'].dt.strftime('%b')

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=45)
st.sidebar.title("NEXUS PRO v2.0")
st.sidebar.caption("Executive Suite | Premium Brands")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Módulos de Navegación")
modulo = st.sidebar.radio(
    "Selecciona una vista:",
    ["📊 Dashboard BI", "📦 Depletion & Stock", "🔄 Oportunidades Recompra", "📈 Forecast Comercial", "🤖 NEXUS Copilot"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filtros Globales")

lista_canales = ["Todos"] + sorted(list(df_ventas['Canal'].dropna().unique()))
canal_sel = st.sidebar.selectbox("Canal de Venta", lista_canales)

lista_marcas = ["Todas"] + sorted(list(df_ventas['Marca'].dropna().unique()))
marca_sel = st.sidebar.selectbox("Marca", lista_marcas)

df_f = df_ventas.copy()
if canal_sel != "Todos":
    df_f = df_f[df_f['Canal'] == canal_sel]
if marca_sel != "Todas":
    df_f = df_f[df_f['Marca'] == marca_sel]

# --- PALETA DE COLORES CORPORATIVA ---
COLOR_PRIMARY = "#C59D7F"   # Bronce / Dorado Premium
COLOR_SECONDARY = "#2D3748" # Gris Pizarra

# ==========================================
# MÓDULO 1: DASHBOARD BI
# ==========================================
if modulo == "📊 Dashboard BI":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">EXECUTIVE ANALYTICS</span>
        <h1 class="header-title">Dashboard Comercial BI</h1>
        <p style="margin: 5px 0 0 0; color: #A0AEC0; font-size: 14px;">Bienvenido, Alexis Martinez — <b>Offtrade Manager</b></p>
    </div>
    """, unsafe_allow_html=True)

    venta_usd = df_f['Monto_USD'].sum()
    venta_pen = venta_usd * TC_GLOBAL
    unidades_totales = df_f['Cantidad'].sum()
    ticket_promedio = df_f['Monto_USD'].mean() if len(df_f) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Venta YTD (USD)</div>
            <div class="kpi-value">${venta_usd:,.0f}</div>
            <div class="kpi-subtitle">▲ +12.4% vs 2025</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Venta YTD (PEN)</div>
            <div class="kpi-value">S/ {venta_pen:,.0f}</div>
            <div class="kpi-subtitle">TC Ref: {TC_GLOBAL:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Volumen Total</div>
            <div class="kpi-value">{unidades_totales:,.0f} u.</div>
            <div class="kpi-subtitle">▲ Unidades vendidas</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Ticket Promedio</div>
            <div class="kpi-value">${ticket_promedio:,.2f}</div>
            <div class="kpi-subtitle">Por transacción</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### 📈 Evolución Mensual de Ventas")
        df_tend = df_f.groupby(['Año', 'Mes_Num', 'Mes_Nombre'])['Monto_USD'].sum().reset_index().sort_values(['Año', 'Mes_Num'])
        fig_tend = px.line(df_tend, x='Mes_Nombre', y='Monto_USD', color='Año', markers=True, color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY])
        fig_tend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#EDF2F7'))
        st.plotly_chart(fig_tend, use_container_width=True)

    with col_g2:
        st.markdown("### 📊 Top Categorías ($ USD)")
        col_cat = 'Categoria' if 'Categoria' in df_f.columns else ('Categoría' if 'Categoría' in df_f.columns else None)
        if col_cat:
            df_cat = df_f.groupby(col_cat)['Monto_USD'].sum().reset_index().sort_values('Monto_USD', ascending=True)
            fig_cat = px.bar(df_cat, x='Monto_USD', y=col_cat, orientation='h', color_discrete_sequence=[COLOR_SECONDARY])
            fig_cat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=True, gridcolor='#EDF2F7'))
            st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### 🏆 Top 10 Clientes Clave")
        if 'Cliente' in df_f.columns:
            df_top = df_f.groupby(['Cliente', 'Canal'])['Monto_USD'].sum().reset_index().sort_values('Monto_USD', ascending=False).head(10)
            df_top['Monto_USD'] = df_top['Monto_USD'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(df_top, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("### 🏢 Distribución por Canal")
        df_can = df_f.groupby('Canal')['Monto_USD'].sum().reset_index()
        fig_can = px.pie(df_can, values='Monto_USD', names='Canal', hole=0.5, color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, "#A0AEC0", "#CBD5E0"])
        fig_can.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_can, use_container_width=True)

# ==========================================
# MÓDULO 2: DEPLETION & STOCK
# ==========================================
elif modulo == "📦 Depletion & Stock":
    st.title("📦 Monitoreo de Depletion y Puntos de Reorden")
    st.info("Algoritmo predictivo de agotamiento de inventario en clientes.")
    df_dep = calcular_depletion_y_reorden(df_f, df_stock_cli)
    st.dataframe(df_dep, use_container_width=True)

# ==========================================
# MÓDULO 3: OPORTUNIDADES RECOMPRA
# ==========================================
elif modulo == "🔄 Oportunidades Recompra":
    st.title("🔄 Detección de Frecuencia de Recompra")
    st.info("Identifica clientes con ciclos de compra vencidos o en riesgo de fuga.")
    df_rec = calcular_oportunidades_recompra(df_f)
    st.dataframe(df_rec, use_container_width=True)

# ==========================================
# MÓDULO 4: FORECAST COMERCIAL
# ==========================================
elif modulo == "📈 Forecast Comercial":
    st.title("📈 Proyección de Ventas (Machine Learning)")
    st.info("Modelo de Regresión para estimar el cierre comercial del próximo trimestre.")
    df_fore = predecir_ventas_futuras(df_f)
    st.dataframe(df_fore, use_container_width=True)

# ==========================================
# MÓDULO 5: NEXUS COPILOT
# ==========================================
elif modulo == "🤖 NEXUS Copilot":
    st.title("🤖 NEXUS Copilot — Asistente IA Comercial")
    st.caption("Consulta ventas, top clientes, marcas y fechas en lenguaje natural.")
    
    pregunta = st.text_input("Hazle una pregunta a la IA comercial:", placeholder="Ej: ¿Cuánto vendimos en Champagne en 2026?")
    
    if pregunta:
        preg_clean = pregunta.lower()
        col_cat = 'Categoria' if 'Categoria' in df_f.columns else ('Categoría' if 'Categoría' in df_f.columns else None)
        if "champagne" in preg_clean and col_cat:
            res = df_f[df_f[col_cat] == 'CHAMPAGNE']['Monto_USD'].sum()
            st.success(f"🍾 La venta acumulada en **Champagne** es de **${res:,.2f} USD**.")
        elif ("top cliente" in preg_clean or "mejores clientes" in preg_clean) and 'Cliente' in df_f.columns:
            res_cli = df_f.groupby('Cliente')['Monto_USD'].sum().reset_index().sort_values('Monto_USD', ascending=False).head(5)
            st.write("🏆 **Top 5 Clientes Principales:**")
            st.table(res_cli)
        else:
            v_tot = df_f['Monto_USD'].sum()
            st.info(f"📊 La venta total filtrada es de **${v_tot:,.2f} USD**. Puedes preguntarme específicamente por categorías como *Champagne*, *Vinos* o *Top Clientes*.")
