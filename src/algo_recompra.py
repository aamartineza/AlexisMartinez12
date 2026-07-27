import pandas as pd
import numpy as np

def calcular_oportunidades_recompra(df_ventas, dias_holgura=7):
    """
    Identifica clientes atrasados en su ciclo habitual de recompra.
    """
    fecha_max_hist = df_ventas['Fecha'].max()

    # 1. Agrupar facturas por cliente
    df_cliente_fac = df_ventas.groupby(['ID_Cliente', 'Nombre_Cliente', 'Tipo_Cliente', 'ID_Factura', 'Fecha']).agg(
        Monto_Factura=('Monto_Venta', 'sum')
    ).reset_index().sort_values(by=['ID_Cliente', 'Fecha'])

    # 2. Calcular diferencia de días entre facturas consecutivas
    df_cliente_fac['Dias_Entre_Compras'] = df_cliente_fac.groupby('ID_Cliente')['Fecha'].diff().dt.days

    # 3. Métricas por Cliente
    df_resumen = df_cliente_fac.groupby(['ID_Cliente', 'Nombre_Cliente', 'Tipo_Cliente']).agg(
        Ultima_Fecha_Compra=('Fecha', 'max'),
        Frecuencia_Promedio_Dias=('Dias_Entre_Compras', 'mean'),
        Total_Facturas=('ID_Factura', 'nunique'),
        Venta_Historica_Total=('Monto_Factura', 'sum')
    ).reset_index()

    # Si un cliente solo tiene 1 compra, se le asigna frecuencia promedio por defecto de 30 días
    df_resumen['Frecuencia_Promedio_Dias'] = df_resumen['Frecuencia_Promedio_Dias'].fillna(30).round(0)

    # 4. Calcular Días Transcurridos (Recencia)
    df_resumen['Dias_Sin_Comprar'] = (fecha_max_hist - df_resumen['Ultima_Fecha_Compra']).dt.days

    # 5. Días de Atraso vs su ciclo normal
    df_resumen['Dias_Atraso'] = df_resumen['Dias_Sin_Comprar'] - df_resumen['Frecuencia_Promedio_Dias']

    # 6. Estado y Nivel de Alerta
    def clasificar_recompra(row):
        if row['Dias_Atraso'] > dias_holgura:
            return "🔴 Pedido Atrasado (Oportunidad)"
        elif row['Dias_Atraso'] >= -2:
            return "🟡 Próximo a Comprar"
        else:
            return "🟢 Ciclo Al Día"

    df_resumen['Estado_Recompra'] = df_resumen.apply(clasificar_recompra, axis=1)

    return df_resumen.sort_values(by='Dias_Atraso', ascending=False)