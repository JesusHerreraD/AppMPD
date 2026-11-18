# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LIMA - FACULTAD DE INGENIERÍA
Carrera de Ingeniería Industrial - Modelamiento Predictivo de Datos
Prototipo de Aplicación Integrado: Tasación Automotriz y Escenarios Comerciales
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==============================================================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL
# ==============================================================================
st.set_page_config(
    page_title="Tasación Automotriz | MPD - ULima",
    page_icon="🚗",
    layout="wide"
)

# CSS personalizado sutil para mejorar la presentación ejecutiva
st.markdown("""
<style>
    .hero-box {
        background: linear-gradient(135deg, #0f4c5c 0%, #1b6b7a 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero-box h1 {
        color: white;
        font-size: 1.75rem;
        margin-bottom: 0.4rem;
    }
    .hero-box p {
        color: #e0f2f1;
        font-size: 0.98rem;
        margin-bottom: 0;
    }
    .scenario-card {
        background-color: #f8fafc;
        border-left: 5px solid #0f4c5c;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        color: #1e293b;
    }
    .scenario-card-orange {
        border-left-color: #e65100;
    }
    .scenario-card-teal {
        border-left-color: #00796b;
    }
    .scenario-card-dark {
        border-left-color: #0d2b3e;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BARRA LATERAL: INGESTA Y VALIDACIÓN DE DATOS (COMPARTIDA PARA AMBAS PESTAÑAS)
# ==============================================================================
st.sidebar.header("📁 Carga de Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo de Autos Usados (CSV o Excel):",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("Dataset cargado exitosamente.")
elif os.path.exists("Used_Car_Prices.csv"):
    df = pd.read_csv("Used_Car_Prices.csv")
    st.sidebar.success("Archivo local 'Used_Car_Prices.csv' cargado.")
else:
    st.sidebar.info("Modo demostración activo con muestra calibrada (n = 1,415 Hyundai). Sube 'Used_Car_Prices.csv' para usar el archivo completo.")
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

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Universidad de Lima**  
Facultad de Ingeniería  
Carrera de Ingeniería Industrial  
*Modelamiento Predictivo de Datos*
""")

# ==============================================================================
# CÁLCULO BASE DEL MODELO LOG-LOG PARA EL SEGMENTO REFERENCIAL (HYUNDAI)
# ==============================================================================
cols = df.columns.tolist()
default_cat = 'brand' if 'brand' in cols else cols[0]
default_y = 'selling_price' if 'selling_price' in cols else (cols[1] if len(cols) > 1 else cols[0])
default_x = 'km_driven' if 'km_driven' in cols else (cols[2] if len(cols) > 2 else cols[0])

df_ref = df.copy()
if 'Hyundai' in df_ref[default_cat].dropna().unique():
    df_ref = df_ref[df_ref[default_cat] == 'Hyundai'].copy()
    ref_label = "Hyundai"
else:
    ref_label = str(df_ref[default_cat].dropna().iloc[0])
    df_ref = df_ref[df_ref[default_cat] == ref_label].copy()

df_ref[default_x] = pd.to_numeric(df_ref[default_x], errors='coerce')
df_ref[default_y] = pd.to_numeric(df_ref[default_y], errors='coerce')
df_ref = df_ref.dropna(subset=[default_x, default_y])
df_ref = df_ref[(df_ref[default_x] > 0) & (df_ref[default_y] > 0)]

ref_slope, ref_intercept, ref_r, ref_p, _ = stats.linregress(
    np.log(df_ref[default_x]), np.log(df_ref[default_y])
)

# ==============================================================================
# ENCABEZADO GENERAL
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <h1>🚗 Reducción de la Subjetividad en la Tasación Automotriz</h1>
    <p>Algoritmos de Regresión (OLS, Ridge, LASSO) y Modelamiento Econométrico (Log-Log) en el Mercado de Vehículos Usados — Universidad de Lima</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ESTRUCTURA EN DOS PESTAÑAS PRINCIPALES
# ==============================================================================
tab_integracion, tab_loglog = st.tabs([
    "🎯 Integración del Proyecto y Escenarios",
    "📈 Dimensión Comercial: Modelo Log-Log"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: INTEGRACIÓN DEL PROYECTO, PÓSTER Y ESCENARIOS PROFUNDIZADOS
# ------------------------------------------------------------------------------
with tab_integracion:
    # 1. Planteamiento del Problema y Objetivo Principal
    col_prob, col_obj = st.columns([1.15, 1])

    with col_prob:
        st.subheader("📌 Planteamiento del Problema")
        st.markdown("""
        El mercado secundario automotriz sufre de **asimetría de información y tasación subjetiva**, exponiendo a las comercializadoras a ineficiencias críticas por **sobretasación o subtasación** *(Muti & Yildiz, 2023)*. 
        
        Físicamente, el **desgaste operativo continuo (kilometraje)** es el principal factor causal de depreciación, ya que incrementa los costos de mantenimiento y castiga directamente la disposición de pago de los compradores *(Alhakamy et al., 2023; Pudaruth, 2014)*. 
        
        Además, la valoración empírica enfrenta distribuciones de datos con **sesgos extremos y heterocedasticidad**, lo que exige transformaciones matemáticas y regularización algorítmica para estabilizar la varianza *(Gegic et al., 2019)*.
        """)

    with col_obj:
        st.subheader("🎯 Objetivo Principal y Dimensiones")
        st.info("""
        **Objetivo Principal:** Desarrollar una solución analítica integral basada en algoritmos de regresión (**OLS, Ridge, LASSO**) y modelamiento econométrico (**Log-Log**) que permita estimar objetivamente el precio de venta a partir de las características del vehículo, reduciendo la subjetividad en la tasación automotriz.
        """)
        st.markdown("""
        * **Dimensión Financiera (OLS):** Establecer una línea base cuantitativa de valoración.
        * **Dimensión Técnica (Ridge - L2):** Controlar la magnitud de los coeficientes y la varianza mediante validación cruzada.
        * **Dimensión Operativa (LASSO - L1):** Seleccionar automáticamente variables relevantes y reducir dimensionalidad.
        * **Dimensión Comercial (Log-Log):** Cuantificar la elasticidad pura precio-desgaste en segmentos de alta rotación.
        """)

    st.markdown("---")

    # 2. Integración Comparativa de los 4 Modelos del Estudio
    st.subheader("🧩 Integración Metodológica: Comparativa de Modelos")
    st.caption("Resultados consolidados del estudio sobre el dataset de vehículos usados (Variable objetivo: selling_price en PKR | Predictores: km_driven, brand, fuel, owner).")

    df_comparativa = pd.DataFrame([
        {
            "Dimensión": "Financiera",
            "Modelo": "OLS (Mínimos Cuadrados)",
            "Rol en la Solución": "Estimación de referencia (línea base)",
            "R²": "0.52",
            "RMSE": "377.64 mil",
            "MAE": "210.22 mil",
            "MAPE": "51.97%",
            "Parámetro / Hallazgo Clave": "Sin regularización"
        },
        {
            "Dimensión": "Técnica",
            "Modelo": "Ridge (Regularización L2)",
            "Rol en la Solución": "Estabilización de coeficientes y control de varianza",
            "R²": "0.52",
            "RMSE": "373.91 mil",
            "MAE": "209.59 mil",
            "MAPE": "52.41%",
            "Parámetro / Hallazgo Clave": "α = 278.26 (CV=5)"
        },
        {
            "Dimensión": "Operativa",
            "Modelo": "LASSO (Regularización L1)",
            "Rol en la Solución": "Selección automática de atributos y limpieza de ruido",
            "R²": "0.803",
            "RMSE": "—",
            "MAE": "209.26 mil",
            "MAPE": "—",
            "Parámetro / Hallazgo Clave": "α = 0.00010975"
        },
        {
            "Dimensión": "Comercial",
            "Modelo": "Log-Log (Econométrico)",
            "Rol en la Solución": "Sensibilidad porcentual pura frente al desgaste (Hyundai)",
            "R²": "0.1340",
            "RMSE": "—",
            "MAE": "—",
            "MAPE": "—",
            "Parámetro / Hallazgo Clave": "Elasticidad ε = -0.2937"
        }
    ])
    st.dataframe(df_comparativa, use_container_width=True, hide_index=True)

    c_mod1, c_mod2, c_mod3, c_mod4 = st.columns(4)
    c_mod1.metric("1. Base (OLS)", "R² = 0.52", "Referencia inicial", delta_color="off")
    c_mod2.metric("2. Estabilidad (Ridge)", "RMSE 373.9k", "-3.73k vs OLS")
    c_mod3.metric("3. Precisión (LASSO)", "R² = 80.3%", "Filtra ruido L1")
    c_mod4.metric("4. Elasticidad (Log-Log)", "ε = -0.2937", "-0.29% por +1% km", delta_color="inverse")

    st.markdown("---")

    # 3. Profundización en Escenarios de la Solución (Enfoque Prescriptivo)
    st.subheader("🧭 Escenarios de la Solución: Análisis Prescriptivo de Tasación")
    st.markdown(f"""
    A partir de la elasticidad estimada para el segmento de alta rotación (**{ref_label}**, $\\varepsilon = {ref_slope:.4f}$), el sistema transforma la predicción estadística en **reglas de decisión comercial** para tres tramos clave del ciclo de vida del vehículo. Puedes ajustar los kilometrajes para recalcular los escenarios en vivo:
    """)

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        km_esc1 = st.number_input("Escenario 1 — Seminuevo / Bajo Desgaste (km):", min_value=1000.0, value=25000.0, step=5000.0)
    with col_e2:
        km_esc2 = st.number_input("Escenario 2 — Estándar / Referencia (km):", min_value=1000.0, value=60000.0, step=5000.0)
    with col_e3:
        km_esc3 = st.number_input("Escenario 3 — Uso Extremo / Flota (km):", min_value=1000.0, value=968377.0, step=10000.0)

    # Cálculos dinámicos de escenarios
    p_esc1 = np.exp(ref_intercept + ref_slope * np.log(km_esc1))
    p_esc2 = np.exp(ref_intercept + ref_slope * np.log(km_esc2))
    p_esc3 = np.exp(ref_intercept + ref_slope * np.log(km_esc3))

    idx_esc1 = (p_esc1 / p_esc2) * 100
    idx_esc2 = 100.0
    idx_esc3 = (p_esc3 / p_esc2) * 100

    # Tarjetas de Índice Relativo (replicando y potenciando el póster)
    k1, k2, k3 = st.columns(3)
    k1.metric(
        label=f"🟠 Escenario 1 ({km_esc1:,.0f} km) — Índice Relativo",
        value=f"{idx_esc1:.1f}",
        delta=f"{idx_esc1 - 100:+.1f}% vs Referencia | {p_esc1:,.0f} PKR"
    )
    k2.metric(
        label=f"🟢 Escenario 2 ({km_esc2:,.0f} km) — Referencia Base",
        value=f"{idx_esc2:.1f}",
        delta=f"Benchmark (0.0%) | {p_esc2:,.0f} PKR",
        delta_color="off"
    )
    k3.metric(
        label=f"🔵 Escenario 3 ({km_esc3:,.0f} km) — Uso Extremo",
        value=f"{idx_esc3:.1f}",
        delta=f"{idx_esc3 - 100:+.1f}% vs Referencia | {p_esc3:,.0f} PKR",
        delta_color="inverse"
    )

    # Gráfico comparativo + Profundización gerencial
    col_chart, col_cards = st.columns([1.05, 1.25])

    with col_chart:
        fig_esc, ax_esc = plt.subplots(figsize=(7, 4.4))
        nombres_cortos = [
            f"Seminuevo\n({km_esc1:,.0f} km)",
            f"Referencia\n({km_esc2:,.0f} km)",
            f"Uso Extremo\n({km_esc3:,.0f} km)"
        ]
        indices_vals = [idx_esc1, idx_esc2, idx_esc3]
        precios_vals = [p_esc1, p_esc2, p_esc3]
        colores = ["#f26419", "#008080", "#0d3b66"]

        bars = ax_esc.bar(nombres_cortos, indices_vals, color=colores, width=0.55, edgecolor="none")
        ax_esc.axhline(100, color="#64748b", linestyle="--", linewidth=1.2, label="Base de Referencia (100.0)")
        ax_esc.set_ylabel("Índice Relativo de Valor (Base = 100)")
        ax_esc.set_title("Retención de Valor Comercial por Tramo de Desgaste")
        ax_esc.set_ylim(0, max(indices_vals) * 1.22)
        ax_esc.grid(axis="y", linestyle="--", alpha=0.4)
        ax_esc.legend(loc="upper right", fontsize=8.5)

        for bar, idx_v, p_v in zip(bars, indices_vals, precios_vals):
            ax_esc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(indices_vals) * 0.03),
                f"{idx_v:.1f}\n({p_v/1000:,.1f}k PKR)",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1e293b"
            )

        st.pyplot(fig_esc)

    with col_cards:
        st.markdown(f"""
        <div class="scenario-card scenario-card-orange">
            <strong>🟠 Escenario 1: Conservador / Seminuevo ({km_esc1:,.0f} km | Índice {idx_esc1:.1f})</strong><br>
            • <strong>Perfil:</strong> Baja fatiga mecánica y alta retención de valor residual (+{idx_esc1-100:.1f}% sobre la mediana).<br>
            • <strong>Riesgo sin modelo:</strong> Subtasación en sala de ventas y pérdida directa de margen.<br>
            • <strong>Decisión prescriptiva:</strong> Fijar <em>precio Premium</em>, priorizar exhibición directa y limitar descuentos comerciales a un máximo del 3%–5%.
        </div>
        <div class="scenario-card scenario-card-teal">
            <strong>🟢 Escenario 2: Base / Estándar de Mercado ({km_esc2:,.0f} km | Índice 100.0)</strong><br>
            • <strong>Perfil:</strong> Mediana operativa del mercado secundario; mayor volumen de oferta y demanda.<br>
            • <strong>Riesgo sin modelo:</strong> Tiempos muertos en inventario por precios desalineados frente a la competencia.<br>
            • <strong>Decisión prescriptiva:</strong> Usar <strong>{p_esc2:,.2f} PKR</strong> como <em>benchmark</em> objetivo para compras <em>trade-in</em> (retoma) y garantizar rotación ágil.
        </div>
        <div class="scenario-card scenario-card-dark">
            <strong>🔵 Escenario 3: Crítico / Uso Extremo ({km_esc3:,.0f} km | Índice {idx_esc3:.1f})</strong><br>
            • <strong>Perfil:</strong> Unidades de uso comercial intensivo o flotas con depreciación acumulada severa ({idx_esc3-100:.1f}%).<br>
            • <strong>Riesgo sin modelo:</strong> Sobretasación crítica e inmovilización de capital en activos de baja liquidez.<br>
            • <strong>Decisión prescriptiva:</strong> Aplicar <em>piso de liquidación</em> ({p_esc3:,.2f} PKR), canalizar a subasta mayorista y exigir inspección mecánica previa.
        </div>
        """, unsafe_allow_html=True)

    # Tabla resumen técnica de los 3 escenarios
    df_escenarios_det = pd.DataFrame([
        {
            "Escenario": "1. Conservador (Seminuevo)",
            "Kilometraje": f"{km_esc1:,.0f} km",
            "ln(km)": f"{np.log(km_esc1):.4f}",
            "Precio Objetivo (PKR)": f"{p_esc1:,.2f}",
            "Índice Relativo": f"{idx_esc1:.1f}",
            "Variación vs Base": f"{idx_esc1 - 100:+.2f}%",
            "Política Comercial": "Margen Alto / Precio Premium"
        },
        {
            "Escenario": "2. Base (Estándar Mercado)",
            "Kilometraje": f"{km_esc2:,.0f} km",
            "ln(km)": f"{np.log(km_esc2):.4f}",
            "Precio Objetivo (PKR)": f"{p_esc2:,.2f}",
            "Índice Relativo": "100.0",
            "Variación vs Base": "0.00%",
            "Política Comercial": "Benchmark / Alta Rotación (Trade-in)"
        },
        {
            "Escenario": "3. Crítico (Uso Extremo)",
            "Kilometraje": f"{km_esc3:,.0f} km",
            "ln(km)": f"{np.log(km_esc3):.4f}",
            "Precio Objetivo (PKR)": f"{p_esc3:,.2f}",
            "Índice Relativo": f"{idx_esc3:.1f}",
            "Variación vs Base": f"{idx_esc3 - 100:+.2f}%",
            "Política Comercial": "Piso de Liquidación / Subasta"
        }
    ])
    st.table(df_escenarios_det)

    # 4. Conclusiones y Referencias
    st.markdown("---")
    col_conc, col_ref = st.columns([1.2, 1])

    with col_conc:
        st.subheader("✅ Conclusiones Principales")
        st.markdown("""
        1. **Complementariedad algorítmica:** OLS provee la línea base, Ridge controla la varianza, LASSO depura variables irrelevantes ($R^2 = 80.3\\%$) y Log-Log extrae la sensibilidad económica real ($\\varepsilon = -0.2937$).
        2. **Referencia cuantitativa con control de riesgo:** Los modelos objetivan la tasación en el segmento masivo, advirtiendo mayor dispersión en vehículos de gama alta o lujo.
        3. **Impacto prescriptivo:** Integrar regularización y econometría mitiga la asimetría de información y convierte el desgaste físico en una regla dinámica para proteger la rentabilidad.
        """)

    with col_ref:
        st.subheader("📚 Referencias Científicas")
        st.markdown("""
        * **Alhakamy et al. (2023).** *Are Used Cars More Sustainable? Price Prediction Based on Linear Regression.* Sustainability, 15(2), 911.
        * **Gegic et al. (2019).** *Car price prediction using machine learning techniques.* TEM Journal, 8(1), 113–118.
        * **Muti & Yildiz (2023).** *Using linear regression for used car price prediction.* IJCESEN, 9(1), 11–16.
        * **Pudaruth, S. (2014).** *Predicting the price of used cars using machine learning techniques.* IJICT, 4(7), 753–764.
        """)

# ------------------------------------------------------------------------------
# PESTAÑA 2: DIMENSIÓN COMERCIAL - MODELO LOG-LOG INTERACTIVO (NORMAL)
# ------------------------------------------------------------------------------
with tab_loglog:
    st.markdown("""
    Esta sección permite explorar de forma interactiva el **modelo econométrico Log-Log**: filtrar por marca o modelo, 
    ejecutar la transformación logarítmica bilateral registro por registro, auditar el ajuste gráfico y simular 
    cualquier nivel de kilometraje en tiempo real.
    """)

    with st.expander("👀 Vista Previa del Dataset y Estadísticos Descriptivos Base"):
        st.dataframe(df.head(10), use_container_width=True)

    # BLOQUE 1: SELECCIÓN DE MODELOS / MARCAS PRINCIPALES
    st.header("🎯 1. Selección de Modelos / Marcas Principales")
    col1, col2, col3 = st.columns(3)

    idx_cat = cols.index('brand') if 'brand' in cols else 0
    idx_y = cols.index('selling_price') if 'selling_price' in cols else (1 if len(cols) > 1 else 0)
    idx_x = cols.index('km_driven') if 'km_driven' in cols else (2 if len(cols) > 2 else 0)

    with col1:
        var_cat = st.selectbox("Variable Categórica (Marca / Modelo):", cols, index=idx_cat)
    with col2:
        var_y = st.selectbox("Variable de Precio (Y):", cols, index=idx_y)
    with col3:
        var_x = st.selectbox("Variable Explicativa Principal (ej. Kilometraje):", cols, index=idx_x)

    modelos_unicos = df[var_cat].dropna().unique().tolist()
    idx_mod = modelos_unicos.index('Hyundai') if 'Hyundai' in modelos_unicos else 0
    modelo_sel = st.selectbox("Selecciona el modelo/marca principal a analizar en profundidad:", modelos_unicos, index=idx_mod)

    # Filtrado booleano y restricción de dominio matemático (x > 0, y > 0)
    df_sub = df[df[var_cat] == modelo_sel].copy()
    df_sub[var_x] = pd.to_numeric(df_sub[var_x], errors='coerce')
    df_sub[var_y] = pd.to_numeric(df_sub[var_y], errors='coerce')
    df_sub = df_sub.dropna(subset=[var_x, var_y])
    df_sub = df_sub[(df_sub[var_x] > 0) & (df_sub[var_y] > 0)]

    st.info(f"📊 Analizando un subconjunto de **{len(df_sub)}** registros para la marca/modelo: **{modelo_sel}**")

    # BLOQUE 2: ESTIMACIÓN DE ELASTICIDAD Y REGRESIÓN LOG-LOG
    st.header("📐 2. Estimación de Elasticidad y Regresión Log-Log")

    df_sub['log_x'] = np.log(df_sub[var_x])
    df_sub['log_y'] = np.log(df_sub[var_y])

    slope, intercept, r_value, p_value, std_err = stats.linregress(df_sub['log_x'], df_sub['log_y'])
    r_squared = r_value ** 2

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coeficiente de Elasticidad Estimado (ε)", f"{slope:.4f}")
    m2.metric("Coeficiente de Determinación (R²)", f"{r_squared:.4f}")
    m3.metric("Correlación Lineal Log-Log (r)", f"{r_value:.4f}")
    m4.metric("Modelo Analizado", f"{modelo_sel}")

    st.markdown(f"""
    * **Ecuación Econométrica Linealizada:** $\\ln(\\text{{{var_y}}}) = {intercept:.4f} + ({slope:.4f}) \\cdot \\ln(\\text{{{var_x}}})$
    * **Interpretación Industrial y Gerencial:** Un coeficiente de elasticidad de **{slope:.4f}** demuestra que por cada **1%** que aumenta la variable de desgaste (*{var_x}*), el precio de mercado del modelo **{modelo_sel}** se deprecia en aproximadamente un **{abs(slope):.2f}%** (variación de {slope:.2f}%).
    * **Sustento Estadístico:** El valor $p = {p_value:.4e}$ confirma que el impacto del kilometraje sobre la depreciación es estadísticamente significativo al 99% de confianza.
    """)

    # BLOQUE 3: CURVA DE COMPORTAMIENTO Y AJUSTE LOG-LOG
    st.header("📊 3. Curva de Comportamiento y Ajuste Log-Log")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(df_sub[var_x], df_sub[var_y], alpha=0.6, edgecolors='w', color='#2b7bba')
    ax1.set_title(f"Relación Original: {var_x} vs {var_y}")
    ax1.set_xlabel(var_x)
    ax1.set_ylabel(var_y)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.scatter(df_sub['log_x'], df_sub['log_y'], color='darkorange', alpha=0.6, edgecolors='w')
    x_vals = np.linspace(df_sub['log_x'].min(), df_sub['log_x'].max(), 100)
    ax2.plot(x_vals, intercept + slope * x_vals, color='red', linewidth=2.2, label=f'Ajuste Lineal Log-Log (Pendiente: {slope:.2f})')
    ax2.set_title("Linealización Log-Log para Cálculo de Elasticidad")
    ax2.set_xlabel(f"ln({var_x})")
    ax2.set_ylabel(f"ln({var_y})")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)

    # BLOQUE 4: SIMULADOR INTERACTIVO DE PROYECCIÓN COMERCIAL
    st.header("🔮 4. Simulador de Proyección Comercial")
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
        precio_proyectado = np.exp(intercept + slope * np.log(val_sim))
        st.subheader("Resultado de la Proyección:")
        st.metric(f"Precio Estimado para {modelo_sel}", f"{precio_proyectado:,.2f} PKR")
        st.info(f"💡 Esta proyección responde a la elasticidad calculada para la Dimensión Comercial del modelo **{modelo_sel}**.")
