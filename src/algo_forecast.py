import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predecir_ventas_futuras(df_ventas, meses_a_predecir=6):
    """
    Predice la tendencia de ventas para los próximos meses usando Regresión Lineal.
    """
    # 1. Agrupar ventas históricas por mes
    df_temp = df_ventas.copy()
    df_temp['Año_Mes'] = df_temp['Fecha'].dt.to_period('M').dt.to_timestamp()
    
    df_mensual = df_temp.groupby('Año_Mes')['Monto_Venta'].sum().reset_index()
    df_mensual = df_mensual.sort_values(by='Año_Mes')
    df_mensual['Mes_Num'] = np.arange(len(df_mensual))

    if len(df_mensual) < 2:
        return df_mensual, 0  # Si hay muy pocos datos

    # 2. Entrenar el modelo de Machine Learning
    X = df_mensual[['Mes_Num']]
    y = df_mensual['Monto_Venta']
    
    modelo = LinearRegression()
    modelo.fit(X, y)

    # 3. Generar predicciones para los meses futuros
    ult_mes_num = df_mensual['Mes_Num'].max()
    meses_futuros_num = np.arange(ult_mes_num + 1, ult_mes_num + 1 + meses_a_predecir).reshape(-1, 1)
    predicciones = modelo.predict(meses_futuros_num)

    # 4. Construir fechas futuras
    ult_fecha = df_mensual['Año_Mes'].max()
    fechas_futuras = pd.date_range(start=ult_fecha + pd.DateOffset(months=1), periods=meses_a_predecir, freq='MS')

    df_futuro = pd.DataFrame({
        'Año_Mes': fechas_futuras,
        'Monto_Venta': np.maximum(0, predicciones),
        'Tipo': '🔮 Proyección (ML)'
    })

    df_mensual['Tipo'] = '📊 Histórico Real'

    df_resultado = pd.concat([df_mensual[['Año_Mes', 'Monto_Venta', 'Tipo']], df_futuro], ignore_index=True)
    
    # Calcular tasa de crecimiento estimada
    tasa_tendencia = modelo.coef_[0]

    return df_resultado, tasa_tendencia