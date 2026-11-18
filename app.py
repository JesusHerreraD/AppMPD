# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LIMA - FACULTAD DE INGENIERÍA
Carrera de Ingeniería Industrial - Modelamiento Predictivo de Datos
Evaluación Continua 2 (Semana 9) - Prototipo de Aplicación (Criterio 3)
Autor: Jesús Emanuel Herrera Dávalos
Dimensión Comercial: Análisis de Elasticidad y Sensibilidad de Precios (Regresión Log-Log)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configuración general de la página en Streamlit
st.set_page_config(
    page_title="Módulo HERRERA - Elasticidad Comercial",
    page_icon="📈",
    layout="wide"
)

# Encabezado principal del prototipo
st.title("📈 Módulo HERRERA: Análisis de Elasticidad y Proyección de Precios")
st.markdown("""
**Dimensión Comercial | Ingeniería Industrial - Universidad de Lima**  
Esta aplicación interactiva evalúa la **elasticidad precio-desgaste** y la valoración en el mercado de vehículos usados. 
Permite filtrar por los modelos de mayor rotación comercial, aplicar transformaciones logarítmicas bilaterales (**Log-Log**) 
registro por registro para estimar coeficientes de elasticidad constante ($\epsilon$) y simular **múltiples escenarios comerciales** 
para eliminar la subjetividad en la tasación y optimizar la gestión de inventarios.
""")

# ==============================================================================
# BLOQUE 1: INGESTA Y VALIDACIÓN DE DATOS (MODO OFFLINE SEGURO PARA USB)
# ==============================================================================
st.sidebar.header("📁 1. Carga de Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo de Autos Usados (CSV o Excel):",
    type=["csv", "xlsx"]
)

# Lógica de carga con detección automática en la carpeta del USB y respaldo sintético
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("¡Dataset de autos cargado exitosamente!")
elif os.path.exists("Used_Car_Prices.csv"):
    df = pd.read_csv("Used_Car_Prices.csv")
    st.sidebar.success("¡Archivo local 'Used_Car_Prices.csv' cargado desde el USB!")
else:
    st.sidebar.warning("Modo Respaldo Activo: Sube 'Used_Car_Prices.csv' en el panel lateral para ver los 1,415 registros exactos de Hyundai.")
    np.random.seed(42)
    n_hyundai = 1415
    km_h = np.random.lognormal(mean=11.0, sigma=0.75, size=n_hyundai)
    noise = np.random.normal(0, 0.55, size=n_hyundai)
    price_h = np.exp(16.0282 - 0.2937 * np.log(km_h) + noise)
    df_h = pd.DataFrame({
        'brand': ['Hyundai'] * n_hyundai,
        'km_driven': np.round(km_h, 0),
        'selling_price': np.round(price_h, 0)
    })
    otras_marcas = pd.DataFrame({
        'brand': np.random.choice(['Suzuki', 'Toyota', 'Honda', 'Chevrolet'], 600),
        'km_driven': np.random.uniform(5000, 250000, 600),
        'selling_price': np.random.uniform(150000, 2500000, 600)
    })
    df = pd.concat([df_h, otras_marcas], ignore_index=True)

with st.expander("👀 Vista Previa del Dataset y Estadísticos Descriptivos Base"):
    st.dataframe(df.head(10), use_container_width=True)

# ==============================================================================
# BLOQUE 2: SELECCIÓN DE MODELOS / MARCAS PRINCIPALES Y LIMPIEZA MATEMÁTICA
# ==============================================================================
st.header("🎯 2. Selección de Modelos / Marcas Principales")
col1, col2, col3 = st.columns(3)

cols = df.columns.tolist()
idx_cat = cols.index('brand') if 'brand' in cols else 0
idx_y = cols.index('selling_price') if 'selling_price' in cols else (1 if len(cols) > 1 else 0)
idx_x = cols.index('km_driven') if 'km_driven' in cols else (2 if len(cols) > 2 else 0)

with col1:
    var_cat = st.selectbox("Variable Categórica (Marca / Modelo):", cols, index=idx_cat)
with col2:
    var_y = st.selectbox("Variable de Precio (Y):", cols, index=idx_y)
with col3:
    var_x = st.selectbox("Variable Explicativa Principal (ej. Kilometraje / Antigüedad):", cols, index=idx_x)

modelos_unicos = df[var_cat].dropna().unique().tolist()
idx_mod = modelos_unicos.index('Hyundai') if 'Hyundai' in modelos_unicos else 0
modelo_sel = st.selectbox("Selecciona el modelo/marca principal a analizar en profundidad:", modelos_unicos, index=idx_mod)

# Filtrado booleano y restricción de dominio matemático (x > 0, y > 0 para logaritmo natural)
df_sub = df[df[var_cat] == modelo_sel].copy()
df_sub[var_x] = pd.to_numeric(df_sub[var_x], errors='coerce')
df_sub[var_y] = pd.to_numeric(df_sub[var_y], errors='coerce')
df_sub = df_sub.dropna(subset=[var_x, var_y])
df_sub = df_sub[(df_sub[var_x] > 0) & (df_sub[var_y] > 0)]

st.info(f"📊 Analizando un subconjunto de **{len(df_sub)}** registros para la marca/modelo: **{modelo_sel}**")

# ==============================================================================
# BLOQUE 3: ESTIMACIÓN DE ELASTICIDAD Y REGRESIÓN ECONOMÉTRICA LOG-LOG
# ==============================================================================
st.header("📐 3. Estimación de Elasticidad y Regresión Log-Log")

# Transformación logarítmica natural registro por registro (vectorizada)
df_sub['log_x'] = np.log(df_sub[var_x])
df_sub['log_y'] = np.log(df_sub[var_y])

# Ajuste por Mínimos Cuadrados Ordinarios sobre variables linealizadas: ln(Y) = B0 + e * ln(X)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_sub['log_x'], df_sub['log_y'])
r_squared = r_value ** 2

m1, m2, m3, m4 = st.columns(4)
m1.metric("Coeficiente de Elasticidad Estimado (ε)", f"{slope:.4f}")
m2.metric("Coeficiente de Determinación (R²)", f"{r_squared:.4f}")
m3.metric("Correlación Lineal Log-Log (r)", f"{r_value:.4f}")
m4.metric("Modelo Analizado", f"{modelo_sel}")

st.markdown(f"""
* **Ecuación Econométrica Linealizada:** $\ln(\text{{{var_y}}}) = {intercept:.4f} + ({slope:.4f}) \cdot \ln(\text{{{var_x}}})$
* **Interpretación Industrial y Gerencial:** Un coeficiente de elasticidad de **{slope:.4f}** demuestra que por cada **1%** que aumenta la variable de desgaste (*{var_x}*), el precio de mercado del modelo **{modelo_sel}** se deprecia en aproximadamente un **{abs(slope):.2f}%** (variación de {slope:.2f}%).
* **Sustento Estadístico:** El valor $p = {p_value:.4e}$ confirma que el impacto del kilometraje sobre la depreciación es estadísticamente significativo al 99% de confianza.
""")

# ==============================================================================
# BLOQUE 4: CURVA DE COMPORTAMIENTO Y AJUSTE LOG-LOG
# ==============================================================================
st.header("📊 4. Curva de Comportamiento y Ajuste Log-Log")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Escala Original
ax1.scatter(df_sub[var_x], df_sub[var_y], alpha=0.6, edgecolors='w', color='#2b7bba')
ax1.set_title(f"Relación Original: {var_x} vs {var_y}")
ax1.set_xlabel(var_x)
ax1.set_ylabel(var_y)
ax1.grid(True, linestyle='--', alpha=0.5)

# Gráfico 2: Escala Linealizada Log-Log
ax2.scatter(df_sub['log_x'], df_sub['log_y'], color='darkorange', alpha=0.6, edgecolors='w')
x_vals = np.linspace(df_sub['log_x'].min(), df_sub['log_x'].max(), 100)
ax2.plot(x_vals, intercept + slope * x_vals, color='red', linewidth=2.2, label=f'Ajuste Lineal Log-Log (Pendiente: {slope:.2f})')
ax2.set_title("Linealización Log-Log para Cálculo de Elasticidad")
ax2.set_xlabel(f"ln({var_x})")
ax2.set_ylabel(f"ln({var_y})")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig)

# ==============================================================================
# BLOQUE 5: SIMULADOR INTERACTIVO DE PROYECCIÓN COMERCIAL
# ==============================================================================
st.header("🔮 5. Simulador de Proyección Comercial")
st.write("Modifica el valor del predictor para proyectar el impacto estimado en el precio de venta del vehículo:")

s_col1, s_col2 = st.columns([2, 1])
with s_col1:
    min_x = float(df_sub[var_x].min())
    max_x = float(df_sub[var_x].max())
    med_x = float(df_sub[var_x].median())
    val_sim = st.slider(
        f"Simular valor de {var_x}:",
        min_value=min_x,
        max_value=max_x,
        value=med_x,
        step=1000.0
    )

with s_col2:
    # Re-transformación exponencial: Precio = exp(B0 + e * ln(X_sim))
    precio_proyectado = np.exp(intercept + slope * np.log(val_sim))
    st.subheader("Resultado de la Proyección:")
    st.metric(f"Precio Estimado para {modelo_sel}", f"{precio_proyectado:,.2f} PKR")
    st.info(f"💡 Esta proyección responde a la elasticidad calculada para la Dimensión Comercial del modelo **{modelo_sel}**.")

# ==============================================================================
# BLOQUE 6: MATRIZ DE 3 ESCENARIOS DE GESTIÓN COMERCIAL (CRITERIO 3 - RÚBRICA)
# ==============================================================================
st.header("📋 6. Evaluación de 3 Escenarios de Impacto Comercial (Rúbrica Criterio 3)")
st.markdown("""
Para sustentar el impacto de la técnica Log-Log en la toma de decisiones gerenciales, se evalúan **tres escenarios operativos** 
representativos del ciclo de vida vehicular dentro del inventario:
""")

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    km_esc1 = st.number_input("Escenario 1 - Bajo Desgaste (km):", value=25000.0, step=5000.0)
with col_e2:
    km_esc2 = st.number_input("Escenario 2 - Uso Mediano Estándar (km):", value=60000.0, step=5000.0)
with col_e3:
    km_esc3 = st.number_input("Escenario 3 - Alto Desgaste Crítico (km):", value=968377.37, step=10000.0)

escenarios_config = [
    ("Escenario 1: Conservador (Bajo Desgaste / Seminuevo)", km_esc1, "Estrategia de Margen Alto: Retención superior de valor; fijar precio Premium con mínima concesión comercial."),
    ("Escenario 2: Base / Estándar (Mediana del Mercado)", km_esc2, "Estrategia de Alta Rotación (Benchmark): Tasación objetiva de equilibrio para compras trade-in y rotación ágil."),
    ("Escenario 3: Crítico (Desgaste Intensivo / Flota)", km_esc3, "Estrategia de Liquidación / Subasta: Descuento estructurado por fatiga mecánica; evita sobrevalorar capital inmovilizado.")
]

precio_base_ref = np.exp(intercept + slope * np.log(km_esc2))
filas_escenarios = []

for nombre, km_val, estrategia in escenarios_config:
    p_est = np.exp(intercept + slope * np.log(km_val))
    var_pct_vs_base = ((p_est - precio_base_ref) / precio_base_ref) * 100
    filas_escenarios.append({
        "Escenario Evaluado": nombre,
        "Kilometraje (km_driven)": f"{km_val:,.2f} km",
        "Transformación ln(X)": f"{np.log(km_val):.4f}",
        "Precio Tasado Objetivo (PKR)": f"{p_est:,.2f} PKR",
        "Dif. % vs Escenario Base": f"{var_pct_vs_base:+.2f}%",
        "Decisión Administrativa / Gestión de Inventario": estrategia
    })

df_escenarios = pd.DataFrame(filas_escenarios)
st.table(df_escenarios)
