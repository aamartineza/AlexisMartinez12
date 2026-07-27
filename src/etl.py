import pandas as pd
import numpy as np

def generar_datos_prueba():
    np.random.seed(42)
    fechas = pd.date_range(start="2025-01-01", end="2026-06-30", freq="D")
    
    clientes = [
        {"ID_Cliente": "C01", "Nombre_Cliente": "LIMA WINE MERCHANTS S.A.C.", "Tipo_Cliente": "Cliente Final"},
        {"ID_Cliente": "C02", "Nombre_Cliente": "CENCOSUD RETAIL PERU S.A.", "Tipo_Cliente": "Supermercados"},
        {"ID_Cliente": "C03", "Nombre_Cliente": "THE CHEESE CORNER S.A.C.", "Tipo_Cliente": "Horecas"},
        {"ID_Cliente": "C04", "Nombre_Cliente": "FOOD SERVICE INVESTMENT S.A.C.", "Tipo_Cliente": "Distribuidores"}
    ]
    
    productos = [
        {"ID_Producto": "P01", "Nombre_Producto": "SAN PELLEGRINO 750ML", "Marca": "SAN PELLEGRINO", "Categoria": "GASIFICADA", "Precio": 15.0},
        {"ID_Producto": "P02", "Nombre_Producto": "MOËT & CHANDON BRUT", "Marca": "MOËT & CHANDON", "Categoria": "CHAMPAGNE", "Precio": 220.0},
        {"ID_Producto": "P03", "Nombre_Producto": "CHANDON EXTRA BRUT", "Marca": "CHANDON", "Categoria": "ESPUMANTES", "Precio": 65.0},
        {"ID_Producto": "P04", "Nombre_Producto": "ILLY CAFE ESPRESSO 250G", "Marca": "ILLY", "Categoria": "CAFE", "Precio": 45.0},
        {"ID_Producto": "P05", "Nombre_Producto": "MONTES ALPHA CABERNET", "Marca": "MONTES", "Categoria": "VINOS", "Precio": 95.0}
    ]
    
    ventas = []
    for _ in range(2000):
        cli = np.random.choice(clientes)
        prod = np.random.choice(productos)
        fec = np.random.choice(fechas)
        cant = np.random.randint(2, 50)
        
        ventas.append({
            "ID_Factura": f"F-{np.random.randint(10000, 99999)}",
            "Fecha": fec,
            "ID_Cliente": cli["ID_Cliente"],
            "Nombre_Cliente": cli["Nombre_Cliente"],
            "Tipo_Cliente": cli["Tipo_Cliente"],
            "ID_Producto": prod["ID_Producto"],
            "Nombre_Producto": prod["Nombre_Producto"],
            "Marca": prod["Marca"],
            "Categoria": prod["Categoria"],
            "Cantidad": cant,
            "Monto_Venta": cant * prod["Precio"],
            "Costo_Estimado": (cant * prod["Precio"]) * 0.7,
            "ID_Bodega": np.random.choice(["BOD-01 (Principal)", "BOD-02 (Sur)"])
        })
        
    df_ventas = pd.DataFrame(ventas)
    df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'])
    
    df_stock_emp = pd.DataFrame([
        {"ID_Bodega": "BOD-01 (Principal)", "ID_Producto": "P01", "Stock_Actual": 150},
        {"ID_Bodega": "BOD-01 (Principal)", "ID_Producto": "P02", "Stock_Actual": 20},
        {"ID_Bodega": "BOD-01 (Principal)", "ID_Producto": "P03", "Stock_Actual": 450},
        {"ID_Bodega": "BOD-01 (Principal)", "ID_Producto": "P04", "Stock_Actual": 12},
        {"ID_Bodega": "BOD-01 (Principal)", "ID_Producto": "P05", "Stock_Actual": 80},
        {"ID_Bodega": "BOD-02 (Sur)", "ID_Producto": "P01", "Stock_Actual": 50},
        {"ID_Bodega": "BOD-02 (Sur)", "ID_Producto": "P05", "Stock_Actual": 5}
    ])
    
    df_stock_cli = pd.DataFrame([
        {"ID_Cliente": "C01", "ID_Producto": "P01", "Stock_Cliente": 15},
        {"ID_Cliente": "C01", "ID_Producto": "P05", "Stock_Cliente": 3},
        {"ID_Cliente": "C02", "ID_Producto": "P03", "Stock_Cliente": 60}
    ])
    
    return df_ventas, df_stock_emp, df_stock_cli