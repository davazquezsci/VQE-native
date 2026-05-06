# ============================================================
# graficar_ajuste_parabolico_spline.py
# Gráfica VQE para tesis con:
#   1) datos VQE estimados,
#   2) spline suavizado como tendencia global,
#   3) ajuste parabólico local del mínimo,
#   4) delimitación visual de la región usada para la parábola.
#
# La región local para la parábola se determina a partir del mínimo
# de un spline suavizado global. Después, la parábola se dibuja
# en todo el intervalo disponible de R_AB.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

# IMPORTANTE:
# El backend PGF debe activarse ANTES de importar pyplot.
mpl.use("pgf")

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_DATOS = Path("datos selectos")
CARPETA_FIGURAS = Path("figuras")
CARPETA_MIN = Path("min")

CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
CARPETA_MIN.mkdir(parents=True, exist_ok=True)

ARCHIVO = "exp03_zoom_ideal_100000.csv"
NOMBRE_FIGURA = "fig_ajuste_spline_parabolico_zoom_ideal_2_100000"
NOMBRE_RESULTADOS = "ajuste_spline_parabolico_zoom_ideal_2_100000.csv"
NOMBRE_RESUMEN_TXT = "resumen_minimos_spline_parabolico_zoom__2ideal_100000.txt"

# Columnas exactas de tu CSV.
COL_R = "R_AB"
COL_VQE = "E_VQE_sim_id"
COL_HF = "E_HF"
COL_QISKIT = "E_qiskit_tot"
COL_FCI = "E_exacta"

# Límites visuales. Si quieres usar todo el intervalo del archivo, deja XLIM = None.
XLIM = (0.8, 1.1)
YLIM = None

# Guardar archivos.
GUARDAR_PDF = True
GUARDAR_PGF = True
GUARDAR_PNG = False
GUARDAR_RESULTADOS = True


# ============================================================
# 2. CONFIGURACIÓN DEL AJUSTE GLOBAL CON SPLINE
# ============================================================

MOSTRAR_SPLINE_GLOBAL = True

# Más grande = curva más holgada.
# Más chico = curva más pegada a los puntos.
# Valores sugeridos: 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
FACTOR_SUAVIZADO_SPLINE = 1e-6

# Grado del spline. k=3 corresponde a spline cúbico.
GRADO_SPLINE = 3


# ============================================================
# 3. CONFIGURACIÓN DEL AJUSTE PARABÓLICO LOCAL
# ============================================================

# Modo para seleccionar los puntos usados en la parábola.
#
# Opciones:
#   "ventana_spline" -> recomendado: usa una vecindad continua alrededor
#                       del mínimo del spline global.
#   "vecinos_spline" -> usa los N_PUNTOS_AJUSTE puntos más cercanos al
#                       mínimo del spline global.
#   "energia_minima" -> modo anterior: usa los N_PUNTOS_AJUSTE puntos
#                       de menor energía. Se conserva sólo como comparación.
MODO_SELECCION_AJUSTE = "vecinos_spline"

# Semiancho de la ventana local para MODO_SELECCION_AJUSTE = "ventana_spline".
# Por ejemplo, DELTA_R_AJUSTE = 0.04 toma el intervalo
# [R_min_spline - 0.04, R_min_spline + 0.04].
DELTA_R_AJUSTE = 0.04

# Número de puntos usados si MODO_SELECCION_AJUSTE = "vecinos_spline"
# o "energia_minima".
# Ejemplos razonables en el zoom: 15, 20, 25, 30.
N_PUNTOS_AJUSTE = 30

# Si True, marca en la gráfica los puntos usados para el ajuste.
MOSTRAR_PUNTOS_AJUSTE = True

# Si True, marca el mínimo estimado por la parábola.
MOSTRAR_MINIMO_AJUSTE = True

# Si True, delimita con sombreado vertical la región usada para la aproximación parabólica.
MOSTRAR_REGION_AJUSTE = True

# Resolución de las curvas suaves dibujadas en todo el intervalo.
N_PUNTOS_CURVA = 700


# ============================================================
# 4. ESTILO LATEX REAL
# ============================================================

mpl.rcParams.update({
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "text.usetex": True,

    "pgf.preamble": r"""
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}[LetterSpace=0.7]
\usepackage{amsmath}
\usepackage{newpxmath}
\usepackage{mathrsfs}
\usepackage{dutchcal}
\usepackage{physics}
\usepackage{chemformula}
""",

    "figure.figsize": (6, 6 * 0.66),
    "figure.dpi": 150,
    "savefig.dpi": 400,

    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.1,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,

    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.55,

    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.75",

    "savefig.bbox": "tight",
})


# ============================================================
# 5. PALETA SOBRIA
# ============================================================

COLOR_FCI = "#333333"
COLOR_HF = "#A8A8A8"
COLOR_QISKIT = "#A10787"
COLOR_VQE_PTS = "#3296B4"
COLOR_SPLINE = "#B32611"
COLOR_PARABOLA = "#111111"
COLOR_PTS_AJUSTE = "#E09F3E"
COLOR_MINIMO = "#111111"
COLOR_REGION = "#E09F3E"


# ============================================================
# 6. FUNCIONES
# ============================================================

def leer_datos() -> pd.DataFrame:
    ruta = CARPETA_DATOS / ARCHIVO

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]

    columnas_necesarias = [COL_R, COL_VQE, COL_HF, COL_QISKIT, COL_FCI]

    for col in columnas_necesarias:
        if col not in df.columns:
            raise KeyError(
                f"No existe la columna '{col}' en el archivo {ARCHIVO}.\n"
                f"Columnas disponibles:\n{df.columns.tolist()}"
            )

    df = df[columnas_necesarias].dropna()
    df = df.sort_values(COL_R).reset_index(drop=True)

    print("\nArchivo leído correctamente:")
    print(ruta)

    print("\nColumnas usadas:")
    print(df.columns.tolist())

    print("\nPrimeras filas usadas:")
    print(df.head().to_string(index=False))

    return df


def obtener_arrays(df: pd.DataFrame):
    R = df[COL_R].to_numpy(dtype=float)
    E_vqe = df[COL_VQE].to_numpy(dtype=float)
    E_hf = df[COL_HF].to_numpy(dtype=float)
    E_qiskit = df[COL_QISKIT].to_numpy(dtype=float)
    E_fci = df[COL_FCI].to_numpy(dtype=float)

    return R, E_vqe, E_hf, E_qiskit, E_fci


def construir_spline_global(R: np.ndarray, E_vqe: np.ndarray):
    """
    Construye un spline suavizado para representar la tendencia global
    de los datos VQE estimados.

    La curva no tiene que interpolar exactamente los puntos, porque los datos
    VQE tienen fluctuaciones estadísticas asociadas al número finito de shots.

    Devuelve:
        spline: spline definido sobre la variable normalizada Rn.
        R_min, R_scale: parámetros de normalización.
    """

    if len(R) <= GRADO_SPLINE:
        raise ValueError(
            f"No hay suficientes puntos para un spline de grado k={GRADO_SPLINE}."
        )

    R_min = R.min()
    R_scale = R.max() - R.min()

    if np.isclose(R_scale, 0.0):
        raise ValueError("El intervalo de R tiene longitud cero; no puede construirse el spline.")

    Rn = (R - R_min) / R_scale

    # Multiplicar por len(R) hace que el parámetro sea menos dependiente
    # del número total de puntos.
    s = FACTOR_SUAVIZADO_SPLINE * len(R)

    spline = UnivariateSpline(Rn, E_vqe, k=GRADO_SPLINE, s=s)

    return spline, R_min, R_scale


def evaluar_spline_global(spline, R_min: float, R_scale: float, R_eval: np.ndarray):
    """Evalúa el spline global en puntos de R físicos, no normalizados."""
    Rn_eval = (R_eval - R_min) / R_scale
    return spline(Rn_eval)


def curva_spline_global(R: np.ndarray, spline, R_min: float, R_scale: float):
    """Devuelve la curva densa del spline para graficar la tendencia global."""

    if not MOSTRAR_SPLINE_GLOBAL:
        return None, None

    R_dense = np.linspace(R.min(), R.max(), N_PUNTOS_CURVA)
    E_dense = evaluar_spline_global(spline, R_min, R_scale, R_dense)

    return R_dense, E_dense


def seleccionar_puntos_parabola(df: pd.DataFrame, R_min_spline: float):
    """
    Selecciona los puntos usados para el ajuste parabólico.

    El modo recomendado es "ventana_spline": primero se localiza el mínimo
    de la tendencia global spline y luego se toma una vecindad continua
    alrededor de ese punto. Esto evita que puntos bajos aislados, debidos
    a ruido estadístico, definan directamente el ajuste local.
    """

    R = df[COL_R].to_numpy(dtype=float)
    E = df[COL_VQE].to_numpy(dtype=float)

    if MODO_SELECCION_AJUSTE == "ventana_spline":
        if DELTA_R_AJUSTE <= 0:
            raise ValueError("DELTA_R_AJUSTE debe ser positivo.")

        mascara = (R >= R_min_spline - DELTA_R_AJUSTE) & (R <= R_min_spline + DELTA_R_AJUSTE)
        df_ajuste = df.loc[mascara].copy()

        if len(df_ajuste) < 3:
            raise ValueError(
                "La ventana alrededor del mínimo del spline contiene menos de 3 puntos. "
                "Aumenta DELTA_R_AJUSTE o usa otro modo de selección."
            )

    elif MODO_SELECCION_AJUSTE == "vecinos_spline":
        if N_PUNTOS_AJUSTE < 3:
            raise ValueError("N_PUNTOS_AJUSTE debe ser al menos 3 para ajustar una parábola.")
        if N_PUNTOS_AJUSTE > len(df):
            raise ValueError(
                f"N_PUNTOS_AJUSTE={N_PUNTOS_AJUSTE} es mayor que el número de datos "
                f"disponibles: {len(df)}."
            )

        idx_vecinos = np.argsort(np.abs(R - R_min_spline))[:N_PUNTOS_AJUSTE]
        df_ajuste = df.iloc[idx_vecinos].copy()

    elif MODO_SELECCION_AJUSTE == "energia_minima":
        if N_PUNTOS_AJUSTE < 3:
            raise ValueError("N_PUNTOS_AJUSTE debe ser al menos 3 para ajustar una parábola.")
        if N_PUNTOS_AJUSTE > len(df):
            raise ValueError(
                f"N_PUNTOS_AJUSTE={N_PUNTOS_AJUSTE} es mayor que el número de datos "
                f"disponibles: {len(df)}."
            )

        idx_menores = np.argsort(E)[:N_PUNTOS_AJUSTE]
        df_ajuste = df.iloc[idx_menores].copy()

    else:
        raise ValueError(
            "MODO_SELECCION_AJUSTE debe ser 'ventana_spline', "
            "'vecinos_spline' o 'energia_minima'."
        )

    df_ajuste = df_ajuste.sort_values(COL_R).reset_index(drop=True)
    return df_ajuste


def ajustar_parabola_minimo(df: pd.DataFrame, spline, R_min_norm: float, R_scale: float):
    """
    Ajusta E(R) = a R^2 + b R + c en una región local determinada
    a partir del mínimo de la tendencia global spline.

    La parábola se calcula sólo con los puntos seleccionados localmente,
    pero se dibuja sobre todo el intervalo de datos para visualizar su
    comportamiento relativo frente a la curva completa.
    """

    R = df[COL_R].to_numpy(dtype=float)
    E = df[COL_VQE].to_numpy(dtype=float)

    R_dense_spline = np.linspace(R.min(), R.max(), max(N_PUNTOS_CURVA, 2000))
    E_dense_spline = evaluar_spline_global(spline, R_min_norm, R_scale, R_dense_spline)

    idx_min_spline = np.argmin(E_dense_spline)
    R_min_spline = R_dense_spline[idx_min_spline]
    E_min_spline = E_dense_spline[idx_min_spline]

    df_ajuste = seleccionar_puntos_parabola(df, R_min_spline)

    R_ajuste = df_ajuste[COL_R].to_numpy(dtype=float)
    E_ajuste = df_ajuste[COL_VQE].to_numpy(dtype=float)

    coef = np.polyfit(R_ajuste, E_ajuste, deg=2)
    a, b, c = coef

    if np.isclose(a, 0.0):
        raise ValueError(
            "El coeficiente cuadrático a es prácticamente cero; "
            "no hay mínimo parabólico bien definido."
        )

    R_min_ajuste = -b / (2 * a)
    E_min_ajuste = a * R_min_ajuste**2 + b * R_min_ajuste + c

    idx_min_discreto = np.argmin(E)
    R_min_discreto = R[idx_min_discreto]
    E_min_discreto = E[idx_min_discreto]

    # La parábola se dibuja en todo el intervalo de los datos,
    # aunque el ajuste se haya calculado sólo con los puntos seleccionados.
    R_dense = np.linspace(R.min(), R.max(), N_PUNTOS_CURVA)
    E_dense = a * R_dense**2 + b * R_dense + c

    R_region_min = R_ajuste.min()
    R_region_max = R_ajuste.max()

    resultados = {
        "a": a,
        "b": b,
        "c": c,
        "R_min_discreto": R_min_discreto,
        "E_min_discreto": E_min_discreto,
        "R_min_spline": R_min_spline,
        "E_min_spline": E_min_spline,
        "R_min_ajuste": R_min_ajuste,
        "E_min_ajuste": E_min_ajuste,
        "R_region_ajuste_min": R_region_min,
        "R_region_ajuste_max": R_region_max,
        "n_puntos_usados": len(df_ajuste),
        "modo_seleccion_ajuste": MODO_SELECCION_AJUSTE,
        "delta_R_ajuste": DELTA_R_AJUSTE if MODO_SELECCION_AJUSTE == "ventana_spline" else np.nan,
        "ajuste_convexo": bool(a > 0),
    }

    print("\n=================================================")
    print("AJUSTE PARABÓLICO DEL MÍNIMO")
    print("=================================================")
    print(f"Archivo leído              : {CARPETA_DATOS / ARCHIVO}")
    print(f"Columna de distancia       : {COL_R}")
    print(f"Columna de energía         : {COL_VQE}")
    print(f"Modo de selección          : {MODO_SELECCION_AJUSTE}")
    if MODO_SELECCION_AJUSTE == "ventana_spline":
        print(f"Semiancho de ventana       : {DELTA_R_AJUSTE}")
    else:
        print(f"Puntos solicitados         : {N_PUNTOS_AJUSTE}")
    print(f"Puntos usados en el ajuste : {len(df_ajuste)}")
    print(f"Mínimo del spline          : R = {R_min_spline:.8f}, E = {E_min_spline:.12f}")
    print(f"Región aproximada en R     : [{R_region_min:.8f}, {R_region_max:.8f}]")

    print("\nPuntos seleccionados para el ajuste:")
    print(df_ajuste[[COL_R, COL_VQE]].to_string(index=False))

    if a <= 0:
        print("\nADVERTENCIA: el ajuste no es convexo (a <= 0).")
        print("El mínimo parabólico puede no ser físicamente representativo.")

    print("\nCoeficientes del ajuste parabólico:")
    print(f"a = {a:.12f}")
    print(f"b = {b:.12f}")
    print(f"c = {c:.12f}")

    print("\nMínimo discreto:")
    print(f"R_min_discreto = {R_min_discreto:.8f}")
    print(f"E_min_discreto = {E_min_discreto:.12f}")

    print("\nMínimo estimado por ajuste:")
    print(f"R_min_ajuste = {R_min_ajuste:.8f}")
    print(f"E_min_ajuste = {E_min_ajuste:.12f}")

    return R_dense, E_dense, df_ajuste, resultados



def imprimir_resumen_minimos(resultados: dict):
    """Imprime un resumen compacto de los mínimos para copiar a LaTeX."""

    print("\n=================================================")
    print("RESUMEN DE MÍNIMOS")
    print("=================================================")
    print(f"Mínimo discreto     : R = {resultados['R_min_discreto']:.8f} Å, "
          f"E = {resultados['E_min_discreto']:.12f} Eh")
    print(f"Mínimo spline       : R = {resultados['R_min_spline']:.8f} Å, "
          f"E = {resultados['E_min_spline']:.12f} Eh")
    print(f"Mínimo parabólico   : R = {resultados['R_min_ajuste']:.8f} Å, "
          f"E = {resultados['E_min_ajuste']:.12f} Eh")
    print(f"Región del ajuste   : [{resultados['R_region_ajuste_min']:.8f}, "
          f"{resultados['R_region_ajuste_max']:.8f}] Å")
    print(f"Puntos usados       : {resultados['n_puntos_usados']}")
    print(f"Modo de selección   : {resultados['modo_seleccion_ajuste']}")

    print("\nValores listos para LaTeX:")
    print(f"E_{{\\Min,\\mathrm{{disc}}}} = {resultados['E_min_discreto']:.6f}\\,\\mathrm{{E}}_{{\\mathrm{{h}}}}")
    print(f"R_{{\\Min,\\mathrm{{disc}}}} = {resultados['R_min_discreto']:.5f}\\,\\text{{\\AA}}")
    print(f"E_{{\\Min,\\mathrm{{aj}}}} = {resultados['E_min_ajuste']:.6f}\\,\\mathrm{{E}}_{{\\mathrm{{h}}}}")
    print(f"R_{{\\Min,\\mathrm{{aj}}}} = {resultados['R_min_ajuste']:.5f}\\,\\text{{\\AA}}")


def guardar_resumen_txt(resultados: dict):
    """Guarda un archivo de texto con los mínimos calculados."""

    ruta_salida = CARPETA_MIN / NOMBRE_RESUMEN_TXT

    lineas = [
        "RESUMEN DE MÍNIMOS\n",
        "==================\n\n",
        f"archivo_origen: {CARPETA_DATOS / ARCHIVO}\n",
        f"columna_distancia: {COL_R}\n",
        f"columna_energia: {COL_VQE}\n",
        f"modo_seleccion_ajuste: {resultados['modo_seleccion_ajuste']}\n",
        f"n_puntos_usados: {resultados['n_puntos_usados']}\n",
        f"factor_suavizado_spline: {FACTOR_SUAVIZADO_SPLINE}\n",
        f"grado_spline: {GRADO_SPLINE}\n\n",
        "Mínimo discreto\n",
        f"R_min_discreto = {resultados['R_min_discreto']:.12f} Å\n",
        f"E_min_discreto = {resultados['E_min_discreto']:.12f} Eh\n\n",
        "Mínimo del spline\n",
        f"R_min_spline = {resultados['R_min_spline']:.12f} Å\n",
        f"E_min_spline = {resultados['E_min_spline']:.12f} Eh\n\n",
        "Mínimo del ajuste parabólico local\n",
        f"R_min_ajuste = {resultados['R_min_ajuste']:.12f} Å\n",
        f"E_min_ajuste = {resultados['E_min_ajuste']:.12f} Eh\n\n",
        "Coeficientes de la parábola E(R)=aR^2+bR+c\n",
        f"a = {resultados['a']:.12f}\n",
        f"b = {resultados['b']:.12f}\n",
        f"c = {resultados['c']:.12f}\n\n",
        "Región usada para el ajuste\n",
        f"R_region_ajuste_min = {resultados['R_region_ajuste_min']:.12f} Å\n",
        f"R_region_ajuste_max = {resultados['R_region_ajuste_max']:.12f} Å\n",
        f"ajuste_convexo = {resultados['ajuste_convexo']}\n\n",
        "Valores listos para LaTeX\n",
        f"E_{{\\Min,\\mathrm{{disc}}}} = {resultados['E_min_discreto']:.6f}\\,\\mathrm{{E}}_{{\\mathrm{{h}}}}\n",
        f"R_{{\\Min,\\mathrm{{disc}}}} = {resultados['R_min_discreto']:.5f}\\,\\text{{\\AA}}\n",
        f"E_{{\\Min,\\mathrm{{aj}}}} = {resultados['E_min_ajuste']:.6f}\\,\\mathrm{{E}}_{{\\mathrm{{h}}}}\n",
        f"R_{{\\Min,\\mathrm{{aj}}}} = {resultados['R_min_ajuste']:.5f}\\,\\text{{\\AA}}\n",
    ]

    ruta_salida.write_text("".join(lineas), encoding="utf-8")
    print("\nArchivo de resumen generado:")
    print(ruta_salida)

def guardar_resultados(resultados: dict):
    if not GUARDAR_RESULTADOS:
        return

    ruta_salida = CARPETA_MIN / NOMBRE_RESULTADOS

    df_resultado = pd.DataFrame({
        "archivo_origen": [str(CARPETA_DATOS / ARCHIVO)],
        "columna_distancia": [COL_R],
        "columna_energia": [COL_VQE],
        "modo_seleccion_ajuste": [resultados["modo_seleccion_ajuste"]],
        "n_puntos_ajuste_configurado": [N_PUNTOS_AJUSTE],
        "n_puntos_usados": [resultados["n_puntos_usados"]],
        "delta_R_ajuste": [resultados["delta_R_ajuste"]],
        "factor_suavizado_spline": [FACTOR_SUAVIZADO_SPLINE],
        "grado_spline": [GRADO_SPLINE],
        "a": [resultados["a"]],
        "b": [resultados["b"]],
        "c": [resultados["c"]],
        "R_region_ajuste_min": [resultados["R_region_ajuste_min"]],
        "R_region_ajuste_max": [resultados["R_region_ajuste_max"]],
        "R_min_discreto": [resultados["R_min_discreto"]],
        "E_min_discreto": [resultados["E_min_discreto"]],
        "R_min_spline": [resultados["R_min_spline"]],
        "E_min_spline": [resultados["E_min_spline"]],
        "R_min_ajuste": [resultados["R_min_ajuste"]],
        "E_min_ajuste": [resultados["E_min_ajuste"]],
        "ajuste_convexo": [resultados["ajuste_convexo"]],
    })

    df_resultado.to_csv(ruta_salida, index=False)
    print("\nArchivo de resultados generado:")
    print(ruta_salida)


def graficar():
    df = leer_datos()
    R, E_vqe, E_hf, E_qiskit, E_fci = obtener_arrays(df)

    spline, R_min_norm, R_scale = construir_spline_global(R, E_vqe)
    R_spline, E_spline = curva_spline_global(R, spline, R_min_norm, R_scale)
    R_parabola, E_parabola, df_ajuste, resultados = ajustar_parabola_minimo(
        df, spline, R_min_norm, R_scale
    )

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Región usada para la aproximación parabólica.
    # --------------------------------------------------------

    if MOSTRAR_REGION_AJUSTE:
        ax.axvspan(
            resultados["R_region_ajuste_min"],
            resultados["R_region_ajuste_max"],
            color=COLOR_REGION,
            alpha=0.13,
            linewidth=0,
            label=r"Región usada para el ajuste parabólico",
            zorder=1,
        )

        ax.axvline(
            resultados["R_region_ajuste_min"],
            color=COLOR_REGION,
            alpha=0.55,
            linestyle=":",
            linewidth=0.9,
            zorder=2,
        )

        ax.axvline(
            resultados["R_region_ajuste_max"],
            color=COLOR_REGION,
            alpha=0.55,
            linestyle=":",
            linewidth=0.9,
            zorder=2,
        )

    # --------------------------------------------------------
    # Spline global: tendencia de los datos VQE.
    # --------------------------------------------------------

    if R_spline is not None:
        ax.plot(
            R_spline,
            E_spline,
            color=COLOR_SPLINE,
            alpha=1,
            linestyle="-",
            linewidth=1.55,
            label=r"VQE ideal, tendencia global spline",
            zorder=8,
        )

    # --------------------------------------------------------
    # Ajuste parabólico local: calculado cerca del mínimo,
    # dibujado en todo el intervalo.
    # --------------------------------------------------------

    ax.plot(
        R_parabola,
        E_parabola,
        color=COLOR_PARABOLA,
        alpha=1,
        linestyle="--",
        linewidth=1.35,
        label=rf"VQE ideal, ajuste parabólico local ($n={resultados['n_puntos_usados']}$)",
        zorder=9,
    )

    # --------------------------------------------------------
    # VQE: todos los datos estimados.
    # --------------------------------------------------------

    ax.scatter(
        R,
        E_vqe,
        s=17,
        marker="o",
        color=COLOR_VQE_PTS,
        alpha=1,
        linewidths=0.1,
        label=r"VQE ideal, datos estimados",
        zorder=7,
    )

    # --------------------------------------------------------
    # Puntos usados para el ajuste parabólico.
    # --------------------------------------------------------

    if MOSTRAR_PUNTOS_AJUSTE:
        ax.scatter(
            df_ajuste[COL_R].to_numpy(dtype=float),
            df_ajuste[COL_VQE].to_numpy(dtype=float),
            s=28,
            marker="x",
            color=COLOR_PTS_AJUSTE,
            alpha=1,
            linewidths=1.0,
            label=r"Puntos usados en la parábola",
            zorder=10,
        )

    # --------------------------------------------------------
    # Mínimo estimado por la parábola.
    # --------------------------------------------------------

    if MOSTRAR_MINIMO_AJUSTE:
        ax.scatter(
            [resultados["R_min_ajuste"]],
            [resultados["E_min_ajuste"]],
            s=34,
            marker="D",
            color=COLOR_MINIMO,
            alpha=1,
            linewidths=0.6,
            label=r"Mínimo parabólico",
            zorder=11,
        )

    # --------------------------------------------------------
    # Referencias.
    # --------------------------------------------------------

    ax.plot(
        R,
        E_hf,
        color=COLOR_HF,
        linestyle="--",
        linewidth=1.55,
        label=r"Hartree--Fock",
        zorder=4,
    )

    ax.plot(
        R,
        E_qiskit,
        color=COLOR_QISKIT,
        alpha=1,
        linestyle="-.",
        linewidth=1.45,
        label=r"Qiskit",
        zorder=3,
    )

    ax.plot(
        R,
        E_fci,
        color=COLOR_FCI,
        linestyle="-",
        linewidth=1.85,
        label=r"FCI",
        zorder=5,
    )

    # --------------------------------------------------------
    # Etiquetas.
    # --------------------------------------------------------

    ax.set_xlabel(r"$R_{AB}$ (Å)")
    ax.set_ylabel(r"$E\,(\mathrm{E}_\mathrm{h})$")

    if XLIM is not None:
        ax.set_xlim(*XLIM)

    if YLIM is not None:
        ax.set_ylim(*YLIM)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="best")
    fig.tight_layout()

    # --------------------------------------------------------
    # Guardado.
    # --------------------------------------------------------

    if GUARDAR_PDF:
        ruta_pdf = CARPETA_FIGURAS / f"{NOMBRE_FIGURA}.pdf"
        fig.savefig(ruta_pdf)
        print(f"Guardado: {ruta_pdf}")

    if GUARDAR_PGF:
        ruta_pgf = CARPETA_FIGURAS / f"{NOMBRE_FIGURA}.pgf"
        fig.savefig(ruta_pgf)
        print(f"Guardado: {ruta_pgf}")

    if GUARDAR_PNG:
        ruta_png = CARPETA_FIGURAS / f"{NOMBRE_FIGURA}.png"
        fig.savefig(ruta_png)
        print(f"Guardado: {ruta_png}")

    imprimir_resumen_minimos(resultados)
    guardar_resultados(resultados)
    guardar_resumen_txt(resultados)

    plt.show()


# ============================================================
# 7. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    graficar()
