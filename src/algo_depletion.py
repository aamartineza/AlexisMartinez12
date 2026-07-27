import pandas as pd
import numpy as np

def calcular_depletion_y_reorden(df_ventas, df_stock_emp, dias_lead_time=45):
    """
    Calcula el ritmo de agotamiento (run-rate), días de stock y alertas de importación.
    """
    # 1. Obtener periodo activo de ventas
    fecha_max = df_ventas['Fecha'].max()
    fecha_min = df_ventas['Fecha'].min()
    dias_totales = max((fecha_max - fecha_min).days, 1)

    # 2. Agrupar ventas totales por producto
    ventas_prod = df_ventas.groupby(['ID_Producto', 'Nombre_Producto', 'Marca', 'Categoria']).agg(
        Total_Unidades_Vendidas=('Cantidad', 'sum'),
        Venta_Total_Soles=('Monto_Venta', 'sum')
    ).reset_index()

    # 3. Calcular Venta Diaria Promedio (Run-Rate)
    ventas_prod['Venta_Diaria_Promedio'] = ventas_prod['Total_Unidades_Vendidas'] / dias_totales

    # 4. Agrupar stock disponible de la empresa por producto
    stock_total = df_stock_emp.groupby('ID_Producto')['Stock_Actual'].sum().reset_index()

    # 5. Unir Ventas + Stock
    df_depletion = pd.merge(ventas_prod, stock_total, on='ID_Producto', how='left')
    df_depletion['Stock_Actual'] = df_depletion['Stock_Actual'].fillna(0)

    # 6. Calcular Días de Inventario Restante (DDI)
    df_depletion['Dias_Stock_Restante'] = np.where(
        df_depletion['Venta_Diaria_Promedio'] > 0,
        df_depletion['Stock_Actual'] / df_depletion['Venta_Diaria_Promedio'],
        999
    )

    # 7. Sugerido de Importación (cobertura meta de 90 días)
    dias_cobertura_meta = 90
    df_depletion['Sugerido_Importacion_Unid'] = np.maximum(
        0, 
        (df_depletion['Venta_Diaria_Promedio'] * dias_cobertura_meta) - df_depletion['Stock_Actual']
    ).round(0)

    # 8. Estado y Alerta de Riesgo
    def clasificar_estado(row):
        if row['Stock_Actual'] == 0:
            return "🔴 Agotado (Sin Stock)"
        elif row['Dias_Stock_Restante'] <= dias_lead_time:
            return "🔴 Crítico (Reorden Urgente)"
        elif row['Dias_Stock_Restante'] <= (dias_lead_time + 30):
            return "🟡 Riesgo Medio"
        else:
            return "🟢 Cobertura Óptima"

    df_depletion['Estado_Riesgo'] = df_depletion.apply(clasificar_estado, axis=1)

    return df_depletion.sort_values(by='Dias_Stock_Restante', ascending=True)