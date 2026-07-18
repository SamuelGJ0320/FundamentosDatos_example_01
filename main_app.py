"""
Dashboard interactivo de riesgo meteorológico — Medellín y Área Metropolitana.
Datos 100% sintéticos, generados dentro de la aplicación.
Ejecutar con: streamlit run main_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Configuración general de la página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Riesgo Meteorológico — Medellín y Área Metropolitana",
    page_icon="🌦️",
    layout="wide",
)

CODIGO_ACCESO = "4650"

# --------------------------------------------------------------------------
# Catálogo de comunas / municipios (Medellín + Área Metropolitana)
# Se asigna una población de referencia y un factor base de susceptibilidad
# a fenómenos de remoción en masa (terreno montañoso -> mayor riesgo)
# --------------------------------------------------------------------------
COMUNAS = {
    "Popular":            {"poblacion": 129_500, "factor_riesgo": 0.85},
    "Santa Cruz":         {"poblacion": 106_600, "factor_riesgo": 0.80},
    "Manrique":           {"poblacion": 157_700, "factor_riesgo": 0.70},
    "Aranjuez":           {"poblacion": 152_400, "factor_riesgo": 0.55},
    "Castilla":           {"poblacion": 154_300, "factor_riesgo": 0.45},
    "Robledo":            {"poblacion": 190_400, "factor_riesgo": 0.65},
    "Villa Hermosa":      {"poblacion": 141_800, "factor_riesgo": 0.75},
    "Buenos Aires":       {"poblacion": 132_600, "factor_riesgo": 0.60},
    "La Candelaria":      {"poblacion": 92_500,  "factor_riesgo": 0.35},
    "Laureles-Estadio":   {"poblacion": 119_100, "factor_riesgo": 0.20},
    "La América":         {"poblacion": 92_700,  "factor_riesgo": 0.40},
    "San Javier":         {"poblacion": 132_000, "factor_riesgo": 0.78},
    "El Poblado":         {"poblacion": 137_200, "factor_riesgo": 0.30},
    "Guayabal":           {"poblacion": 95_500,  "factor_riesgo": 0.35},
    "Belén":              {"poblacion": 195_300, "factor_riesgo": 0.42},
    "Bello":              {"poblacion": 482_400, "factor_riesgo": 0.68},
    "Itagüí":             {"poblacion": 292_600, "factor_riesgo": 0.50},
    "Envigado":           {"poblacion": 239_800, "factor_riesgo": 0.38},
    "Sabaneta":           {"poblacion": 62_500,  "factor_riesgo": 0.33},
    "La Estrella":        {"poblacion": 76_200,  "factor_riesgo": 0.55},
    "Copacabana":         {"poblacion": 82_100,  "factor_riesgo": 0.58},
    "Girardota":          {"poblacion": 55_600,  "factor_riesgo": 0.50},
    "Barbosa":            {"poblacion": 48_300,  "factor_riesgo": 0.47},
    "Caldas":             {"poblacion": 96_700,  "factor_riesgo": 0.62},
}

DIRECCIONES_VIENTO = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
NIVELES_RIESGO = ["Bajo", "Medio", "Alto", "Rojo"]

NUMERIC_COLS = [
    "Temperatura_C",
    "Humedad_Relativa",
    "Velocidad_Viento_kmh",
    "Precipitacion_mm",
    "Poblacion",
    "Indice_Calidad_Aire",
]
CATEGORICAL_COLS = ["Comuna", "Direccion_Viento", "Nivel_Riesgo"]
DATE_COL = "Fecha"


# --------------------------------------------------------------------------
# Generación de datos sintéticos
# --------------------------------------------------------------------------
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    """Crea un DataFrame sintético con 10 columnas de tipos de datos variados,
    simulando condiciones meteorológicas por comuna del Valle de Aburrá."""
    rng = np.random.default_rng(semilla)

    nombres_comunas = list(COMUNAS.keys())
    comuna_sel = rng.choice(nombres_comunas, size=n_registros)

    fecha_inicio = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
    fechas = fecha_inicio + pd.to_timedelta(rng.integers(0, 365, size=n_registros), unit="D")

    # Clima tropical de montaña: temperaturas moderadas, humedad alta, lluvias frecuentes
    temperatura = np.round(rng.normal(23, 3.2, size=n_registros), 1)
    humedad = np.round(np.clip(rng.normal(72, 12, size=n_registros), 30, 100), 1)
    viento = np.round(np.clip(rng.gamma(2.0, 4.0, size=n_registros), 0, 60), 1)
    direccion_viento = rng.choice(DIRECCIONES_VIENTO, size=n_registros)
    precipitacion = np.round(np.clip(rng.gamma(1.3, 12.0, size=n_registros), 0, 180), 1)
    ica = np.round(np.clip(rng.normal(75, 25, size=n_registros), 5, 250), 1)

    poblacion = np.array([COMUNAS[c]["poblacion"] for c in comuna_sel])
    factor_riesgo_base = np.array([COMUNAS[c]["factor_riesgo"] for c in comuna_sel])

    # Puntaje de riesgo combinado: pondera lluvia, viento, humedad y susceptibilidad del terreno
    puntaje_riesgo = (
        0.45 * (precipitacion / 180)
        + 0.15 * (viento / 60)
        + 0.15 * (humedad / 100)
        + 0.25 * factor_riesgo_base
    )
    nivel_riesgo = pd.cut(
        puntaje_riesgo,
        bins=[-0.01, 0.35, 0.55, 0.75, 1.01],
        labels=NIVELES_RIESGO,
    ).astype(str)

    df = pd.DataFrame(
        {
            "Fecha": fechas,
            "Comuna": comuna_sel,
            "Temperatura_C": temperatura,
            "Humedad_Relativa": humedad,
            "Velocidad_Viento_kmh": viento,
            "Direccion_Viento": direccion_viento,
            "Precipitacion_mm": precipitacion,
            "Poblacion": poblacion,
            "Indice_Calidad_Aire": ica,
            "Nivel_Riesgo": nivel_riesgo,
        }
    )
    return df.sort_values("Fecha").reset_index(drop=True)


# --------------------------------------------------------------------------
# Panel lateral institucional
# --------------------------------------------------------------------------
def mostrar_panel_institucional():
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0; border-bottom: 2px solid #002855; margin-bottom:1rem;">
            <h2 style="color:#002855; margin-bottom:0;">EAFIT 2026</h2>
            <p style="margin:0; font-size:0.95rem; color:#333;"><b>Ciencia de Datos</b></p>
            <p style="margin:0; font-size:0.9rem; color:#333;">Profesor Jorge Padilla</p>
            <p style="margin:0; font-size:0.85rem; color:#666;">Julio de 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


mostrar_panel_institucional()

# --------------------------------------------------------------------------
# Autenticación con código de acceso
# --------------------------------------------------------------------------
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.sidebar.subheader("🔒 Acceso al dashboard")
    codigo_ingresado = st.sidebar.text_input("Código de acceso", type="password")
    if st.sidebar.button("Ingresar", use_container_width=True):
        if codigo_ingresado == CODIGO_ACCESO:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.sidebar.error("Código incorrecto. Intenta nuevamente.")

    st.title("🌦️ Dashboard de Riesgo Meteorológico")
    st.info("🔐 Ingresa el código de acceso en el panel izquierdo para continuar.")
    st.stop()

# --------------------------------------------------------------------------
# A partir de aquí, contenido protegido por el código de acceso
# --------------------------------------------------------------------------
if "semilla" not in st.session_state:
    st.session_state.semilla = 42
if "df" not in st.session_state:
    st.session_state.df = generar_datos(500, st.session_state.semilla)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Simulación de datos")

n_registros = st.sidebar.slider("Número de registros", min_value=100, max_value=1500, value=500, step=50)
nueva_semilla = st.sidebar.number_input(
    "Semilla aleatoria", min_value=0, max_value=9999, value=st.session_state.semilla, step=1
)

if st.sidebar.button("🔄 Generar nuevos datos", use_container_width=True):
    st.session_state.semilla = nueva_semilla
    st.session_state.df = generar_datos(n_registros, nueva_semilla)

df = st.session_state.df

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros")
comunas_sel = st.sidebar.multiselect(
    "Comuna / Municipio", options=sorted(df["Comuna"].unique()), default=sorted(df["Comuna"].unique())
)
rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(df["Fecha"].min().date(), df["Fecha"].max().date()),
)
niveles_sel = st.sidebar.multiselect(
    "Nivel de riesgo", options=NIVELES_RIESGO, default=NIVELES_RIESGO
)

df_filtrado = df[df["Comuna"].isin(comunas_sel) & df["Nivel_Riesgo"].isin(niveles_sel)]
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    inicio, fin = rango_fechas
    df_filtrado = df_filtrado[
        (df_filtrado["Fecha"].dt.date >= inicio) & (df_filtrado["Fecha"].dt.date <= fin)
    ]

st.sidebar.download_button(
    "⬇️ Descargar datos (CSV)",
    data=df_filtrado.to_csv(index=False).encode("utf-8"),
    file_name="datos_meteorologicos_medellin.csv",
    mime="text/csv",
    use_container_width=True,
)

if st.sidebar.button("🔓 Cerrar sesión", use_container_width=True):
    st.session_state.autenticado = False
    st.rerun()

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("🌦️ Dashboard de Riesgo Meteorológico — Medellín y Área Metropolitana")
st.caption(
    "Datos sintéticos generados dentro de la aplicación, con fines académicos y de simulación "
    "para apoyar la toma de decisiones frente a posibles riesgos o desastres por comuna/municipio."
)

with st.expander("📄 Ver tabla de datos generados", expanded=False):
    st.dataframe(df_filtrado, use_container_width=True)

if df_filtrado.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

# --------------------------------------------------------------------------
# Tabs principales
# --------------------------------------------------------------------------
tab_cuanti, tab_cuali, tab_graf, tab_riesgo = st.tabs(
    ["📊 Estadística cuantitativa", "🗂️ Estadística cualitativa", "📈 Análisis gráfico", "🚨 Panel de riesgo"]
)

# ------------------------- TAB 1: Cuantitativa -----------------------------
with tab_cuanti:
    st.subheader("Resumen de variables numéricas")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temp. promedio", f'{df_filtrado["Temperatura_C"].mean():.1f} °C')
    col2.metric("Humedad promedio", f'{df_filtrado["Humedad_Relativa"].mean():.1f} %')
    col3.metric("Precipitación total", f'{df_filtrado["Precipitacion_mm"].sum():,.0f} mm')
    col4.metric("Población en zonas filtradas", f'{df_filtrado["Poblacion"].sum():,}')

    st.markdown("#### Estadísticos descriptivos")
    resumen = df_filtrado[NUMERIC_COLS].describe().T
    resumen["mediana"] = df_filtrado[NUMERIC_COLS].median()
    resumen = resumen.rename(
        columns={
            "mean": "media", "std": "desv_std", "min": "mínimo", "max": "máximo",
            "25%": "percentil_25", "50%": "percentil_50", "75%": "percentil_75",
        }
    )
    st.dataframe(resumen.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Matriz de correlación")
    corr = df_filtrado[NUMERIC_COLS].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre variables meteorológicas y poblacionales",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------- TAB 2: Cualitativa -------------------------------
with tab_cuali:
    st.subheader("Distribución de variables categóricas")

    c1, c2 = st.columns(2)
    with c1:
        conteo_riesgo = df_filtrado["Nivel_Riesgo"].value_counts().reindex(NIVELES_RIESGO).reset_index()
        conteo_riesgo.columns = ["Nivel_Riesgo", "Registros"]
        fig_riesgo = px.bar(
            conteo_riesgo, x="Nivel_Riesgo", y="Registros", color="Nivel_Riesgo",
            color_discrete_map={"Bajo": "#2E7D32", "Medio": "#F9A825", "Alto": "#EF6C00", "Rojo": "#C62828"},
            title="Número de registros por nivel de riesgo",
        )
        st.plotly_chart(fig_riesgo, use_container_width=True)

    with c2:
        conteo_viento = df_filtrado["Direccion_Viento"].value_counts().reset_index()
        conteo_viento.columns = ["Direccion_Viento", "Registros"]
        fig_viento = px.pie(
            conteo_viento, names="Direccion_Viento", values="Registros",
            title="Distribución de la dirección del viento", hole=0.4,
        )
        st.plotly_chart(fig_viento, use_container_width=True)

    st.markdown("#### Registros por comuna / municipio")
    conteo_comuna = df_filtrado["Comuna"].value_counts().reset_index()
    conteo_comuna.columns = ["Comuna", "Registros"]
    fig_comuna = px.bar(
        conteo_comuna.sort_values("Registros"), x="Registros", y="Comuna", orientation="h",
        title="Número de registros por comuna/municipio",
    )
    fig_comuna.update_layout(height=600)
    st.plotly_chart(fig_comuna, use_container_width=True)

    st.markdown("#### Moda y frecuencia relativa")
    moda_comuna = df_filtrado["Comuna"].mode()[0]
    pct_riesgo_alto = (df_filtrado["Nivel_Riesgo"].isin(["Alto", "Rojo"]).sum() / len(df_filtrado)) * 100
    m1, m2 = st.columns(2)
    m1.metric("Comuna más frecuente en los registros", moda_comuna)
    m2.metric("% de registros con riesgo Alto/Rojo", f"{pct_riesgo_alto:.1f}%")

# ------------------------- TAB 3: Análisis gráfico --------------------------
with tab_graf:
    st.subheader("Explorador gráfico interactivo")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        tipo_grafica = st.selectbox(
            "Tipo de gráfica",
            ["Línea", "Barras", "Dispersión (Scatter)", "Histograma", "Boxplot"],
        )
    with ctrl2:
        var_x = st.selectbox(
            "Variable eje X", options=[DATE_COL] + CATEGORICAL_COLS + NUMERIC_COLS, index=0,
        )
    with ctrl3:
        var_y = st.selectbox("Variable eje Y", options=NUMERIC_COLS, index=0)

    ctrl4, ctrl5, ctrl6 = st.columns(3)
    with ctrl4:
        color_por = st.selectbox("Agrupar/color por", options=["Ninguno"] + CATEGORICAL_COLS, index=1)
    with ctrl5:
        paleta = st.selectbox("Paleta de color", ["Plotly", "Vivid", "Bold", "Pastel", "Set2"], index=0)
    with ctrl6:
        mostrar_umbral = st.checkbox("Mostrar línea de umbral", value=True)

    umbral_valor = None
    if mostrar_umbral:
        umbral_valor = st.slider(
            f"Valor de umbral para '{var_y}'",
            min_value=float(df_filtrado[var_y].min()),
            max_value=float(df_filtrado[var_y].max()),
            value=float(df_filtrado[var_y].mean()),
        )

    color_arg = None if color_por == "Ninguno" else color_por
    paletas = {
        "Plotly": px.colors.qualitative.Plotly,
        "Vivid": px.colors.qualitative.Vivid,
        "Bold": px.colors.qualitative.Bold,
        "Pastel": px.colors.qualitative.Pastel,
        "Set2": px.colors.qualitative.Set2,
    }

    df_plot = df_filtrado.sort_values(by=var_x) if var_x in df_filtrado.columns else df_filtrado

    if tipo_grafica == "Línea":
        fig = px.line(
            df_plot, x=var_x, y=var_y, color=color_arg, markers=True,
            color_discrete_sequence=paletas[paleta], title=f"{var_y} en función de {var_x}",
        )
    elif tipo_grafica == "Barras":
        fig = px.bar(
            df_plot, x=var_x, y=var_y, color=color_arg, color_discrete_sequence=paletas[paleta],
            barmode="group", title=f"{var_y} por {var_x}",
        )
    elif tipo_grafica == "Dispersión (Scatter)":
        fig = px.scatter(
            df_plot, x=var_x, y=var_y, color=color_arg, size=var_y,
            color_discrete_sequence=paletas[paleta], title=f"{var_y} vs {var_x}",
        )
    elif tipo_grafica == "Histograma":
        fig = px.histogram(
            df_plot, x=var_y, color=color_arg, color_discrete_sequence=paletas[paleta],
            title=f"Distribución de {var_y}",
        )
    else:  # Boxplot
        fig = px.box(
            df_plot, x=color_arg if color_arg else var_x, y=var_y, color=color_arg,
            color_discrete_sequence=paletas[paleta], title=f"Distribución de {var_y}",
        )

    if mostrar_umbral and umbral_valor is not None and tipo_grafica != "Histograma":
        fig.add_hline(
            y=umbral_valor, line_dash="dash", line_color="red",
            annotation_text=f"Umbral: {umbral_valor:.2f}", annotation_position="top left",
        )

    fig.update_layout(legend_title_text=color_por if color_arg else "")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Comparación múltiple de variables (normalizada)")
    vars_comparar = st.multiselect(
        "Selecciona variables numéricas a comparar en el tiempo",
        options=NUMERIC_COLS, default=["Precipitacion_mm", "Velocidad_Viento_kmh"],
    )
    if vars_comparar:
        df_norm = df_filtrado.sort_values("Fecha").copy()
        for col in vars_comparar:
            rango = df_norm[col].max() - df_norm[col].min()
            df_norm[col + "_norm"] = (df_norm[col] - df_norm[col].min()) / rango if rango != 0 else 0
        fig_multi = go.Figure()
        for col in vars_comparar:
            fig_multi.add_trace(
                go.Scatter(x=df_norm["Fecha"], y=df_norm[col + "_norm"], mode="lines+markers", name=col)
            )
        fig_multi.update_layout(
            title="Comparación normalizada (0-1) en el tiempo", yaxis_title="Valor normalizado", xaxis_title="Fecha",
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Selecciona al menos una variable para comparar.")

# ------------------------- TAB 4: Panel de riesgo ---------------------------
with tab_riesgo:
    st.subheader("Panel de apoyo a la decisión — Riesgo por comuna/municipio")

    umbral_precip = st.slider("Umbral crítico de precipitación (mm)", 0, 180, 80)
    umbral_viento = st.slider("Umbral crítico de velocidad de viento (km/h)", 0, 60, 35)

    df_alerta = df_filtrado[
        (df_filtrado["Precipitacion_mm"] >= umbral_precip)
        | (df_filtrado["Velocidad_Viento_kmh"] >= umbral_viento)
    ]

    resumen_comuna = (
        df_filtrado.groupby("Comuna")
        .agg(
            Poblacion=("Poblacion", "first"),
            Precipitacion_media=("Precipitacion_mm", "mean"),
            Viento_medio=("Velocidad_Viento_kmh", "mean"),
            Registros_riesgo_alto=("Nivel_Riesgo", lambda s: s.isin(["Alto", "Rojo"]).sum()),
            Total_registros=("Nivel_Riesgo", "size"),
        )
        .reset_index()
    )
    resumen_comuna["Pct_riesgo_alto"] = (
        resumen_comuna["Registros_riesgo_alto"] / resumen_comuna["Total_registros"] * 100
    ).round(1)
    resumen_comuna["Poblacion_expuesta_estimada"] = (
        resumen_comuna["Poblacion"] * resumen_comuna["Pct_riesgo_alto"] / 100
    ).round(0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Registros que superan umbrales", len(df_alerta))
    m2.metric("Comunas con eventos críticos", df_alerta["Comuna"].nunique())
    m3.metric(
        "Población potencialmente expuesta (estimada)",
        f'{int(resumen_comuna["Poblacion_expuesta_estimada"].sum()):,}',
    )

    fig_pob_riesgo = px.bar(
        resumen_comuna.sort_values("Pct_riesgo_alto", ascending=False),
        x="Comuna", y="Pct_riesgo_alto", color="Pct_riesgo_alto",
        color_continuous_scale="YlOrRd",
        title="% de registros en riesgo Alto/Rojo por comuna/municipio",
        labels={"Pct_riesgo_alto": "% Riesgo Alto/Rojo"},
    )
    st.plotly_chart(fig_pob_riesgo, use_container_width=True)

    fig_burbujas = px.scatter(
        resumen_comuna, x="Precipitacion_media", y="Viento_medio", size="Poblacion",
        color="Pct_riesgo_alto", color_continuous_scale="YlOrRd", hover_name="Comuna",
        title="Relación precipitación / viento / población por comuna (tamaño = población)",
        labels={"Precipitacion_media": "Precipitación media (mm)", "Viento_medio": "Viento medio (km/h)"},
    )
    st.plotly_chart(fig_burbujas, use_container_width=True)

    st.markdown("#### Tabla resumen para toma de decisiones")
    st.dataframe(
        resumen_comuna.sort_values("Pct_riesgo_alto", ascending=False).style.format(
            {
                "Precipitacion_media": "{:.1f}",
                "Viento_medio": "{:.1f}",
                "Pct_riesgo_alto": "{:.1f}%",
                "Poblacion": "{:,.0f}",
                "Poblacion_expuesta_estimada": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "Dashboard construido con Streamlit + Plotly · Datos 100% sintéticos generados en tiempo real · "
    "EAFIT 2026 · Ciencia de Datos · Profesor Jorge Padilla"
)