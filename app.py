import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os

from src.etl import generar_datos_prueba
from src.algo_depletion import calcular_depletion_y_reorden
from src.algo_recompra import calcular_oportunidades_recompra
from src.algo_forecast import predecir_ventas_futuras

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Premium Brands | Dashboard Comercial",
    layout="wide",
    page_icon="⭐",
    initial_sidebar_state="expanded"
)

TC_GLOBAL = 3.50

# --- CARGA Y PREPARACIÓN DE DATOS DESDE PARQUET / FALLBACK ---
@st.cache_data(ttl=3600)
def obtener_datos(file=None):
    if file is not None:
        # Carga manual por el usuario en el sidebar
        if file.name.endswith('.parquet'):
            df = pd.read_parquet(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Mapeo de columnas dinámico
        col_fecha = [c for c in df.columns if 'fecha' in c.lower() or 'fec' in c.lower()][0]
        col_monto = [c for c in df.columns if 'monto' in c.lower() or 'venta' in c.lower() or 'total' in c.lower()][0]
        
        df['Fecha'] = pd.to_datetime(df[col_fecha], errors='coerce')
        df['Monto_Venta'] = pd.to_numeric(df[col_monto], errors='coerce').fillna(0)
        return df, None, None

    # Intentar leer el histórico Parquet generado localmente
    PARQUET_HISTORICO = 'ventas_historico_2020_2026.parquet'
    if os.path.exists(PARQUET_HISTORICO):
        df = pd.read_parquet(PARQUET_HISTORICO)
        
        # Limpieza de nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Detección inteligente de columnas clave
        cols_fec = [c for c in df.columns if 'fecha' in c.lower() or 'fec' in c.lower()]
        cols_mon = [c for c in df.columns if 'monto' in c.lower() or 'venta' in c.lower() or 'total' in c.lower() or 'vta' in c.lower()]
        cols_cli = [c for c in df.columns if 'cliente' in c.lower() or 'razon' in c.lower() or 'nom' in c.lower()]
        cols_mar = [c for c in df.columns if 'marca' in c.lower()]
        cols_cat = [c for c in df.columns if 'cat' in c.lower() or 'fam' in c.lower()]
        cols_can = [c for c in df.columns if 'canal' in c.lower() or 'tipo' in c.lower()]

        # Asignación estandarizada
        df['Fecha'] = pd.to_datetime(df[cols_fec[0]], errors='coerce') if cols_fec else pd.to_datetime('2026-01-01')
        df['Monto_Venta'] = pd.to_numeric(df[cols_mon[0]], errors='coerce').fillna(0) if cols_mon else 0.0
        df['Nombre_Cliente'] = df[cols_cli[0]].astype(str) if cols_cli else 'Desconocido'
        df['Marca'] = df[cols_mar[0]].astype(str) if cols_mar else 'General'
        df['Categoria'] = df[cols_cat[0]].astype(str) if cols_cat else 'General'
        df['Tipo_Cliente'] = df[cols_can[0]].astype(str) if cols_can else 'Offtrade'
        df['Cantidad'] = pd.to_numeric(df['Cantidad'], errors='coerce').fillna(1) if 'Cantidad' in df.columns else 1

        # Intento de lectura opcional de Stocks
        df_stock_emp = pd.read_parquet('stockxlotes28.07.parquet') if os.path.exists('stockxlotes28.07.parquet') else None
        df_stock_cli = None
        
        return df, df_stock_emp, df_stock_cli

    # Si no hay Parquet histórico, usa generador de datos de prueba
    return generar_datos_prueba()

# Carga Inicial de Datos
df_ventas, df_stock_emp, df_stock_cli = obtener_datos(None)

# Creación de Columnas de Tiempo y Moneda
df_ventas['Año'] = df_ventas['Fecha'].dt.year
df_ventas['Mes_Num'] = df_ventas['Fecha'].dt.month
df_ventas['Día_Num'] = df_ventas['Fecha'].dt.day
df_ventas['Mes_Nombre'] = df_ventas['Fecha'].dt.strftime('%b').str.lower()
df_ventas['Venta_USD'] = df_ventas['Monto_Venta'] / TC_GLOBAL

# --- CÁLCULO DE MÉTRICAS HEADER ---
def calcular_metricas_header(df):
    fecha_max = df['Fecha'].max() if not df.empty else pd.Timestamp('2026-07-29')
    año_actual = fecha_max.year
    dia_del_año = fecha_max.dayofyear
    
    df_hoy = df[df['Fecha'].dt.date == fecha_max.date()]
    venta_hoy_usd = df_hoy['Venta_USD'].sum()
    
    df_ytd_act = df[(df['Año'] == año_actual) & (df['Fecha'].dt.dayofyear <= dia_del_año)]
    ytd_actual_usd = df_ytd_act['Venta_USD'].sum()
    
    df_ytd_ant = df[(df['Año'] == año_actual - 1) & (df['Fecha'].dt.dayofyear <= dia_del_año)]
    ytd_anterior_usd = df_ytd_ant['Venta_USD'].sum()
    
    var_ytd_pct = ((ytd_actual_usd - ytd_anterior_usd) / ytd_anterior_usd) * 100 if ytd_anterior_usd > 0 else 0.0
    
    str_fecha = fecha_max.strftime('%d/%m/%Y')
    str_hora = fecha_max.strftime('%H:%M:%S') if fecha_max.time() != pd.Timestamp('00:00:00').time() else "23:59:59"
    
    return ytd_actual_usd, var_ytd_pct, venta_hoy_usd, str_fecha, str_hora

ytd_usd, var_pct, hoy_usd, fecha_cierre, hora_cierre = calcular_metricas_header(df_ventas)
color_var = '#38A169' if var_pct >= 0 else '#E53E3E'
symbol_var = '▲' if var_pct >= 0 else '▼'

# --- ENCABEZADO Y GARANTÍA TOTAL DEL BOTÓN DEL SIDEBAR ---
st.html(f"""
<style>
.stAppDeployButton, #MainMenu, [data-testid="stAppDeployButton"] {{
    display: none !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 1000001 !important;
}}
[data-testid="stSidebarCollapseButton"], 
button[data-testid="baseButton-header"],
[data-testid="collapsedControl"] {{
    z-index: 1000005 !important;
    position: fixed !important;
    top: 12px !important;
    left: 15px !important;
    display: flex !important;
    visibility: visible !important;
}}
.main .block-container {{
    padding-top: 105px !important;
    background-color: #FFFFFF;
}}
html, body, [data-testid="stAppViewContainer"] {{
    background-color: #FFFFFF;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}}
.sticky-header {{
    position: fixed;
    top: 0;
    right: 0;
    left: 0;
    height: 85px;
    background-color: #FFFFFF;
    z-index: 99999;
    padding: 8px 30px 8px 80px;
    border-bottom: 1.5px solid #C59D7F;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}
.brand-group {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}}
.star-logo {{
    color: #C59D7F;
    font-size: 24px;
    line-height: 1;
    margin-bottom: 0px;
}}
.brand-title {{
    font-size: 22px;
    font-weight: 300;
    color: #4A5568;
    margin: 0;
    line-height: 1;
    letter-spacing: 0.5px;
    white-space: nowrap;
}}
.subtitle-offtrade {{
    font-size: 12px;
    color: #C59D7F;
    font-weight: 600;
    margin-top: 2px;
    letter-spacing: 0.5px;
    white-space: nowrap;
}}
.main-title-header {{
    font-size: 19px;
    color: #4A5568;
    font-weight: 300;
    margin-left: 20px;
    border-left: 1px solid #E2E8F0;
    padding-left: 20px;
    line-height: 1.2;
    white-space: nowrap;
}}
.metrics-group {{
    display: flex;
    align-items: center;
    gap: 25px;
}}
.metric-card-header {{
    display: flex;
    flex-direction: column;
    text-align: right;
}}
.label-header {{
    font-size: 10px;
    color: #A0AEC0;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.8px;
    white-space: nowrap;
}}
.val-header {{
    font-size: 18px;
    font-weight: 700;
    color: #2D3748;
    line-height: 1.1;
    white-space: nowrap;
}}
.badge-header {{
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
}}
.divider-header {{
    border-left: 1px solid #E2E8F0;
    height: 40px;
}}
.tag-header {{
    font-size: 11px;
    color: #C59D7F;
    font-weight: 700;
    text-align: right;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    line-height: 1.2;
    white-space: nowrap;
}}
</style>

<div class="sticky-header">
    <div style="display: flex; align-items: center;">
        <div class="brand-group">
            <div class="star-logo">★</div>
            <div class="brand-title">Premium Brands</div>
            <div class="subtitle-offtrade">Offtrade Manager</div>
        </div>
        <div class="main-title-header">
            Dashboard Comercial BI<br>
            <span style="font-size: 13px; color: #C59D7F; font-weight: 500;">Bienvenido Alexis</span>
        </div>
    </div>
    
    <div class="metrics-group">
        <div class="metric-card-header">
            <span class="label-header">Venta YTD 2026 ($ USD)</span>
            <span class="val-header">$ {ytd_usd:,.0f}</span>
            <span class="badge-header" style="color: {color_var};">
                {symbol_var} {abs(var_pct):.1f}% vs 2025 YTD
            </span>
        </div>
        <div class="divider-header"></div>
        <div class="metric-card-header">
            <span class="label-header">Venta Hoy Facturada ($ USD)</span>
            <span class="val-header">$ {hoy_usd:,.2f}</span>
            <span class="label-header" style="color: #C59D7F; font-size: 10px; margin-top: 1px;">
                Cierre: {fecha_cierre}<br>
                <span style="color: #718096; font-weight: 700;">{hora_cierre}</span>
            </span>
        </div>
        <div class="divider-header"></div>
        <div class="tag-header">DASHBOARD<br>COMERCIAL</div>
    </div>
</div>
""")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.title("Filtros Globales")

archivo_subido = st.sidebar.file_uploader("Actualizar base de ventas:", type=["parquet", "xlsx", "csv"])
if archivo_subido is not None:
    df_ventas, _, _ = obtener_datos(archivo_subido)
    df_ventas['Año'] = df_ventas['Fecha'].dt.year
    df_ventas['Mes_Num'] = df_ventas['Fecha'].dt.month
    df_ventas['Mes_Nombre'] = df_ventas['Fecha'].dt.strftime('%b').str.lower()
    df_ventas['Venta_USD'] = df_ventas['Monto_Venta'] / TC_GLOBAL

años_disp = sorted(df_ventas['Año'].dropna().unique())
años_sel = st.sidebar.multiselect("Filtrar por Año:", años_disp, default=años_disp)

canales_disp = sorted(df_ventas['Tipo_Cliente'].dropna().unique())
canales_sel = st.sidebar.multiselect("Filtrar por Canal:", canales_disp, default=canales_disp)

df_f = df_ventas[(df_ventas['Año'].isin(años_sel)) & (df_ventas['Tipo_Cliente'].isin(canales_sel))]

st.sidebar.markdown("---")
modulo = st.sidebar.radio(
    "Selecciona un Módulo:",
    ["📊 Dashboard BI", "🚨 Depletion", "🎯 Recompra", "🔮 Forecast", "🤖 NEXUS Copilot"]
)

# -----------------------------------------------------------------------------
# MÓDULO 1: DASHBOARD BI
# -----------------------------------------------------------------------------
if modulo == "📊 Dashboard BI":
    m1, m2, m3 = st.columns(3)
    m1.metric("Venta Total ($ USD)", f"$ {df_f['Venta_USD'].sum():,.2f}")
    m2.metric("Unidades Vendidas", f"{df_f['Cantidad'].sum():,.0f}")
    m3.metric("Ticket Promedio", f"$ {df_f['Venta_USD'].mean():,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Evolución Mensual ($ USD)**")
        df_evol = df_f.groupby(['Mes_Num', 'Mes_Nombre', 'Año'])['Venta_USD'].sum().reset_index().sort_values('Mes_Num')
        fig1 = px.line(df_evol, x='Mes_Nombre', y='Venta_USD', color='Año', markers=True, color_discrete_sequence=['#C59D7F', '#4A5568'])
        fig1.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        st.markdown("**2. Top 10 Categorías ($ USD)**")
        df_cat = df_f.groupby('Categoria')['Venta_USD'].sum().reset_index().sort_values('Venta_USD', ascending=True).tail(10)
        fig2 = px.bar(df_cat, x='Venta_USD', y='Categoria', orientation='h', color_discrete_sequence=['#C59D7F'])
        fig2.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**3. Distribución por Canal**")
        df_can = df_f.groupby('Tipo_Cliente')['Venta_USD'].sum().reset_index()
        fig3 = px.pie(df_can, values='Venta_USD', names='Tipo_Cliente', hole=0.5, color_discrete_sequence=['#C59D7F', '#4A5568', '#A0AEC0'])
        fig3.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    
    with c4:
        st.markdown("**4. Top 10 Marcas ($ USD)**")
        df_mar = df_f.groupby('Marca')['Venta_USD'].sum().reset_index().sort_values('Venta_USD', ascending=False).head(10)
        fig4 = px.bar(df_mar, x='Marca', y='Venta_USD', color_discrete_sequence=['#4A5568'])
        fig4.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**5. Top 10 Clientes**")
        df_cli = df_f.groupby('Nombre_Cliente')['Venta_USD'].sum().reset_index().sort_values('Venta_USD', ascending=False).head(10)
        st.dataframe(df_cli.style.format({"Venta_USD": "$ {:,.2f}"}), use_container_width=True, height=220, hide_index=True)
    
    with c6:
        st.markdown("**6. Matriz Categoría vs Canal ($ USD)**")
        pivot = df_f.pivot_table(index='Categoria', columns='Tipo_Cliente', values='Venta_USD', aggfunc='sum', fill_value=0)
        st.dataframe(pivot.style.format("$ {:,.0f}"), use_container_width=True, height=220)

# -----------------------------------------------------------------------------
# MÓDULO 2: DEPLETION
# -----------------------------------------------------------------------------
elif modulo == "🚨 Depletion":
    st.markdown("<h3 style='color: #4A5568; font-weight: 300;'>🚨 Depletion & Reorden</h3>", unsafe_allow_html=True)
    lead_time = st.sidebar.slider("Lead Time (Días):", 15, 90, 45, step=5)
    df_dep = calcular_depletion_y_reorden(df_ventas, df_stock_emp, dias_lead_time=lead_time)
    st.dataframe(df_dep, use_container_width=True, height=400)

# -----------------------------------------------------------------------------
# MÓDULO 3: RECOMPRA
# -----------------------------------------------------------------------------
elif modulo == "🎯 Recompra":
    st.markdown("<h3 style='color: #4A5568; font-weight: 300;'>🎯 Matriz de Recompra</h3>", unsafe_allow_html=True)
    dias_h = st.sidebar.slider("Días Tolerancia:", 1, 20, 7)
    df_rec = calcular_oportunidades_recompra(df_ventas, dias_holgura=dias_h)
    st.dataframe(df_rec, use_container_width=True, height=400)

# -----------------------------------------------------------------------------
# MÓDULO 4: FORECAST
# -----------------------------------------------------------------------------
elif modulo == "🔮 Forecast":
    st.markdown("<h3 style='color: #4A5568; font-weight: 300;'>🔮 Forecast Predictivo</h3>", unsafe_allow_html=True)
    meses_p = st.sidebar.slider("Meses a Predecir:", 1, 12, 6)
    df_fore, _ = predecir_ventas_futuras(df_ventas, meses_a_predecir=meses_p)
    fig_f = px.line(df_fore, x='Año_Mes', y='Monto_Venta', color='Tipo', markers=True)
    st.plotly_chart(fig_f, use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO 5: NEXUS COPILOT
# -----------------------------------------------------------------------------
elif modulo == "🤖 NEXUS Copilot":
    st.markdown("<h3 style='color: #4A5568; font-weight: 300;'>🤖 NEXUS Copilot</h3>", unsafe_allow_html=True)
    st.caption("Procesador en lenguaje natural con soporte para Clientes, Rangos de Meses, Años, Marcas, Días y Ordenamiento.")
    
    pregunta = st.text_input("Ingresa tu consulta libre:", value="dame la venta a cencosud de enero a marzo en el 2026, ordena los registros por factura")

    if pregunta:
        q = pregunta.lower().strip()
        df_res = df_ventas.copy()
        filtros = []

        # 1. FILTRO DE CLIENTE
        clientes_disp = df_ventas['Nombre_Cliente'].dropna().unique()
        for cli in clientes_disp:
            if cli.lower() in q or any(p in q for p in cli.lower().split() if len(p) > 3):
                df_res = df_res[df_res['Nombre_Cliente'].str.contains(cli, case=False, na=False)]
                filtros.append(f"Cliente: **{cli}**")
                break

        # 2. FILTRO DE MARCA
        marcas_disp = df_ventas['Marca'].dropna().unique()
        for mar in marcas_disp:
            if mar.lower() in q:
                df_res = df_res[df_res['Marca'].str.contains(mar, case=False, na=False)]
                filtros.append(f"Marca: **{mar}**")
                break

        # 3. FILTRO DE AÑO
        años_enc = re.findall(r'\b(202[0-6])\b', q)
        if años_enc:
            ano_sel = int(años_enc[0])
            df_res = df_res[df_res['Año'] == ano_sel]
            filtros.append(f"Año: **{ano_sel}**")

        # 4. FILTRO DE RANGO DE MESES O MES ÚNICO
        meses_dict = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
            "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
        }
        
        rango_meses = re.search(r'de\s+([a-z]+)\s+a\s+([a-z]+)', q)
        if rango_meses and rango_meses.group(1) in meses_dict and rango_meses.group(2) in meses_dict:
            m_inicio = meses_dict[rango_meses.group(1)]
            m_fin = meses_dict[rango_meses.group(2)]
            df_res = df_res[(df_res['Mes_Num'] >= m_inicio) & (df_res['Mes_Num'] <= m_fin)]
            filtros.append(f"Rango Meses: **{rango_meses.group(1).capitalize()} - {rango_meses.group(2).capitalize()}**")
        else:
            for m_nombre, m_num in meses_dict.items():
                if m_nombre in q:
                    df_res = df_res[df_res['Mes_Num'] == m_num]
                    filtros.append(f"Mes: **{m_nombre.capitalize()}**")
                    break

        # 5. FILTRO DE DÍAS
        p_dias = re.search(r'primeros?\s+(\d+)', q)
        u_dias = re.search(r'últimos?\s+(\d+)|ultimos?\s+(\d+)', q)
        dia_exacto = re.search(r'\b(dia|día|el)?\s*(\d{1,2})\s*(de)?\s*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b', q)

        if p_dias:
            num_d = int(p_dias.group(1))
            df_res = df_res[df_res['Día_Num'] <= num_d]
            filtros.append(f"Días: **Primeros {num_d} días**")
        elif u_dias:
            num_d = int(u_dias.group(1) or u_dias.group(2))
            df_res = df_res[df_res['Día_Num'] >= (31 - num_d)]
            filtros.append(f"Días: **Últimos {num_d} días**")
        elif dia_exacto:
            num_dia = int(dia_exacto.group(2))
            df_res = df_res[df_res['Día_Num'] == num_dia]
            filtros.append(f"Día Exacto: **{num_dia}**")

        # 6. ORDENAMIENTO DE REGISTROS
        col_factura = 'ID_Factura' if 'ID_Factura' in df_res.columns else ('Factura' if 'Factura' in df_res.columns else None)
        
        if "ordena" in q or "ordenar" in q or "ordenados" in q:
            if "factura" in q and col_factura:
                df_res = df_res.sort_values(by=col_factura, ascending=True)
                filtros.append("Orden: **Por N° Factura (A-Z)**")
            elif "fecha" in q:
                df_res = df_res.sort_values(by='Fecha', ascending=True)
                filtros.append("Orden: **Por Fecha (Ascendente)**")
            else:
                df_res = df_res.sort_values(by='Venta_USD', ascending=False)
        else:
            df_res = df_res.sort_values(by='Venta_USD', ascending=False)

        st.markdown("<br>", unsafe_allow_html=True)
        if filtros:
            st.success(f"🎯 **Criterios Detectados:** {' | '.join(filtros)}")

        monto_usd = df_res['Venta_USD'].sum()
        cant_total = df_res['Cantidad'].sum() if 'Cantidad' in df_res.columns else 0
        num_trans = len(df_res)

        # Métricas principales en Dólares ($ USD)
        c1, c2, c3 = st.columns(3)
        c1.metric("Venta Total ($ USD)", f"$ {monto_usd:,.2f}")
        c2.metric("Unidades Vendidas", f"{cant_total:,.0f}")
        c3.metric("N° Registros / Facturas", f"{num_trans}")

        st.markdown("### 📋 Detalle de Ventas ($ USD)")
        
        # Construcción de la tabla
        df_tabla = pd.DataFrame()
        df_tabla['Fecha'] = df_res['Fecha'].dt.strftime('%Y-%m-%d')
        
        if col_factura:
            df_tabla['Factura'] = df_res[col_factura]
        else:
            df_tabla['Factura'] = "F-" + df_res.index.astype(str)

        df_tabla['Cliente'] = df_res['Nombre_Cliente']
        df_tabla['Marca'] = df_res['Marca']
        df_tabla['Producto'] = df_res['Nombre_Producto'] if 'Nombre_Producto' in df_res.columns else df_res['Categoria']
        df_tabla['Cantidad'] = df_res['Cantidad'].astype(int)
        df_tabla['Venta USD'] = df_res['Venta_USD'].apply(lambda x: f"$ {x:,.2f}")

        st.dataframe(df_tabla, use_container_width=True, height=350, hide_index=True)

        csv_data = df_tabla.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Reporte (CSV)", csv_data, "Reporte_Copilot.csv", "text/csv")