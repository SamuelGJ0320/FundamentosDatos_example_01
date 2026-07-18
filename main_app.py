"""
Dashboard interactivo de COVID-19 con datos sintéticos.
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
    page_title="Dashboard COVID-19 (Datos Sintéticos)",
    page_icon="🦠",
    layout="wide",
)

REGIONES = ["Norte", "Sur", "Centro", "Oriente", "Occidente"]

NUMERIC_COLS = [
    "Casos_Confirmados",
    "Casos_Recuperados",
    "Fallecidos",
    "Tasa_Positividad",
    "Ocupacion_UCI",
]
CATEGORICAL_COLS = ["Region", "Campana_Vacunacion"]
DATE_COL = "Fecha"


# --------------------------------------------------------------------------
# Generación de datos sintéticos
# --------------------------------------------------------------------------
def generar_datos(n_registros: int, semilla: int) -> pd.DataFrame:
    """Crea un DataFrame sintético con 8 columnas de tipos de datos variados."""
    rng = np.random.default_rng(semilla)

    fechas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_registros, freq="D")
    region = rng.choice(REGIONES, size=n_registros)

    confirmados = rng.integers(50, 800, size=n_registros)
    # Los recuperados y fallecidos se derivan de los confirmados para mantener coherencia
    recuperados = (confirmados * rng.uniform(0.6, 0.9, size=n_registros)).astype(int)
    fallecidos = (confirmados * rng.uniform(0.01, 0.06, size=n_registros)).astype(int)

    tasa_positividad = np.round(rng.uniform(1.0, 35.0, size=n_registros), 2)  # %
    ocupacion_uci = np.round(rng.uniform(20.0, 100.0, size=n_registros), 2)  # %
    campana_vacunacion = rng.choice([True, False], size=n_registros, p=[0.65, 0.35])

    df = pd.DataFrame(
        {
            "Fecha": fechas,
            "Region": region,
            "Casos_Confirmados": confirmados,
            "Casos_Recuperados": recuperados,
            "Fallecidos": fallecidos,
            "Tasa_Positividad": tasa_positividad,
            "Ocupacion_UCI": ocupacion_uci,
            "Campana_Vacunacion": campana_vacunacion,
        }
    )
    return df


# --------------------------------------------------------------------------
# Estado de la sesión: los datos persisten hasta que el usuario los regenere
# --------------------------------------------------------------------------
if "semilla" not in st.session_state:
    st.session_state.semilla = 42
if "df" not in st.session_state:
    st.session_state.df = generar_datos(10, st.session_state.semilla)

# --------------------------------------------------------------------------
# Barra lateral: controles de simulación
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulación de datos")

n_registros = st.sidebar.slider("Número de registros", min_value=10, max_value=60, value=10, step=1)
nueva_semilla = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=9999, value=st.session_state.semilla, step=1)

if st.sidebar.button("🔄 Generar nuevos datos", use_container_width=True):
    st.session_state.semilla = nueva_semilla
    st.session_state.df = generar_datos(n_registros, nueva_semilla)

df = st.session_state.df

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros")
regiones_sel = st.sidebar.multiselect("Región", options=sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(df["Fecha"].min().date(), df["Fecha"].max().date()),
)

df_filtrado = df[df["Region"].isin(regiones_sel)]
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    inicio, fin = rango_fechas
    df_filtrado = df_filtrado[
        (df_filtrado["Fecha"].dt.date >= inicio) & (df_filtrado["Fecha"].dt.date <= fin)
    ]

st.sidebar.download_button(
    "⬇️ Descargar datos (CSV)",
    data=df_filtrado.to_csv(index=False).encode("utf-8"),
    file_name="covid_datos_sinteticos.csv",
    mime="text/csv",
    use_container_width=True,
)

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("🦠 Dashboard COVID-19 — Datos Sintéticos")
st.caption(
    "Todos los datos mostrados son simulados dentro de la aplicación con fines demostrativos, "
    "no corresponden a cifras oficiales."
)

with st.expander("📄 Ver tabla de datos generados", expanded=False):
    st.dataframe(df_filtrado, use_container_width=True)

if df_filtrado.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

# --------------------------------------------------------------------------
# Tabs principales
# --------------------------------------------------------------------------
tab_cuanti, tab_cuali, tab_graf = st.tabs(
    ["📊 Estadística cuantitativa", "🗂️ Estadística cualitativa", "📈 Análisis gráfico"]
)

# ------------------------- TAB 1: Cuantitativa -----------------------------
with tab_cuanti:
    st.subheader("Resumen de variables numéricas")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total confirmados", int(df_filtrado["Casos_Confirmados"].sum()))
    col2.metric("Total recuperados", int(df_filtrado["Casos_Recuperados"].sum()))
    col3.metric("Total fallecidos", int(df_filtrado["Fallecidos"].sum()))
    col4.metric("Tasa positividad prom.", f'{df_filtrado["Tasa_Positividad"].mean():.2f}%')

    st.markdown("#### Estadísticos descriptivos")
    resumen = df_filtrado[NUMERIC_COLS].describe().T
    resumen["mediana"] = df_filtrado[NUMERIC_COLS].median()
    resumen = resumen.rename(
        columns={
            "mean": "media",
            "std": "desv_std",
            "min": "mínimo",
            "max": "máximo",
            "25%": "percentil_25",
            "50%": "percentil_50",
            "75%": "percentil_75",
        }
    )
    st.dataframe(resumen.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Matriz de correlación")
    corr = df_filtrado[NUMERIC_COLS].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlación entre variables numéricas",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------- TAB 2: Cualitativa -------------------------------
with tab_cuali:
    st.subheader("Distribución de variables categóricas")

    c1, c2 = st.columns(2)
    with c1:
        conteo_region = df_filtrado["Region"].value_counts().reset_index()
        conteo_region.columns = ["Region", "Registros"]
        fig_region = px.bar(
            conteo_region, x="Region", y="Registros", color="Region",
            title="Número de registros por región",
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with c2:
        conteo_vac = df_filtrado["Campana_Vacunacion"].value_counts().reset_index()
        conteo_vac.columns = ["Campana_Activa", "Registros"]
        conteo_vac["Campana_Activa"] = conteo_vac["Campana_Activa"].map({True: "Activa", False: "Inactiva"})
        fig_vac = px.pie(
            conteo_vac, names="Campana_Activa", values="Registros",
            title="Registros con campaña de vacunación activa",
            hole=0.4,
        )
        st.plotly_chart(fig_vac, use_container_width=True)

    st.markdown("#### Moda y frecuencia relativa")
    moda_region = df_filtrado["Region"].mode()[0]
    pct_vac_activa = (df_filtrado["Campana_Vacunacion"].sum() / len(df_filtrado)) * 100
    m1, m2 = st.columns(2)
    m1.metric("Región más frecuente", moda_region)
    m2.metric("% con campaña de vacunación activa", f"{pct_vac_activa:.1f}%")

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
            "Variable eje X",
            options=[DATE_COL] + CATEGORICAL_COLS + NUMERIC_COLS,
            index=0,
        )
    with ctrl3:
        var_y = st.selectbox("Variable eje Y", options=NUMERIC_COLS, index=0)

    ctrl4, ctrl5, ctrl6 = st.columns(3)
    with ctrl4:
        color_por = st.selectbox("Agrupar/color por", options=["Ninguno"] + CATEGORICAL_COLS, index=1)
    with ctrl5:
        paleta = st.selectbox(
            "Paleta de color",
            ["Plotly", "Vivid", "Bold", "Pastel", "Set2"],
            index=0,
        )
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
            color_discrete_sequence=paletas[paleta],
            title=f"{var_y} en función de {var_x}",
        )
    elif tipo_grafica == "Barras":
        fig = px.bar(
            df_plot, x=var_x, y=var_y, color=color_arg,
            color_discrete_sequence=paletas[paleta], barmode="group",
            title=f"{var_y} por {var_x}",
        )
    elif tipo_grafica == "Dispersión (Scatter)":
        fig = px.scatter(
            df_plot, x=var_x, y=var_y, color=color_arg, size=var_y,
            color_discrete_sequence=paletas[paleta],
            title=f"{var_y} vs {var_x}",
        )
    elif tipo_grafica == "Histograma":
        fig = px.histogram(
            df_plot, x=var_y, color=color_arg,
            color_discrete_sequence=paletas[paleta],
            title=f"Distribución de {var_y}",
        )
    else:  # Boxplot
        fig = px.box(
            df_plot, x=color_arg if color_arg else var_x, y=var_y, color=color_arg,
            color_discrete_sequence=paletas[paleta],
            title=f"Distribución de {var_y}",
        )

    if mostrar_umbral and umbral_valor is not None and tipo_grafica != "Histograma":
        fig.add_hline(
            y=umbral_valor,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Umbral: {umbral_valor:.2f}",
            annotation_position="top left",
        )

    fig.update_layout(legend_title_text=color_por if color_arg else "")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Comparación múltiple de variables (normalizada)")
    vars_comparar = st.multiselect(
        "Selecciona variables numéricas a comparar en el tiempo",
        options=NUMERIC_COLS,
        default=["Casos_Confirmados", "Casos_Recuperados"],
    )
    if vars_comparar:
        df_norm = df_filtrado.sort_values("Fecha").copy()
        for col in vars_comparar:
            rango = df_norm[col].max() - df_norm[col].min()
            df_norm[col + "_norm"] = (
                (df_norm[col] - df_norm[col].min()) / rango if rango != 0 else 0
            )
        fig_multi = go.Figure()
        for col in vars_comparar:
            fig_multi.add_trace(
                go.Scatter(
                    x=df_norm["Fecha"], y=df_norm[col + "_norm"],
                    mode="lines+markers", name=col,
                )
            )
        fig_multi.update_layout(
            title="Comparación normalizada (0-1) en el tiempo",
            yaxis_title="Valor normalizado",
            xaxis_title="Fecha",
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Selecciona al menos una variable para comparar.")

st.markdown("---")
st.caption("Dashboard construido con Streamlit + Plotly · Datos 100% sintéticos generados en tiempo real.")