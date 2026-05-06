# ============================================================
# generar_grafica_error_tesis.py
# Gráficas para tesis VQE con estilo LaTeX real:
# Error absoluto respecto a FCI
# XeLaTeX + fontspec + TeX Gyre Pagella
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
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

ARCHIVO = "exp03_zoom_ideal_100000.csv"
NOMBRE_FIGURA = "fig01_error_zoom_ideal_100000"

# Columnas exactas de tu CSV.
# Modifica estos nombres según cada archivo.
COL_R = "R_AB"
COL_VQE = "E_VQE_sim_id"
COL_FCI = "E_exacta"

# Límites visuales.
XLIM = (0.8, 1.1)
YLIM = None

# Guardar archivos.
GUARDAR_PDF = True
GUARDAR_PGF = True
GUARDAR_PNG = False


# ============================================================
# 2. CONFIGURACIÓN DEL AJUSTE DEL ERROR
# ============================================================
# Los errores |E_VQE - E_FCI| se muestran como puntos.
# La línea suave NO interpola forzosamente los puntos.
#
# Opciones:
#   "smoothing_spline"  -> recomendado
#   "polinomio"         -> ajuste global por mínimos cuadrados
#   "ninguno"           -> sólo puntos

METODO_AJUSTE_ERROR = "smoothing_spline"

# Smoothing spline:
# Más grande = curva más holgada.
# Más chico = curva más pegada a los puntos.
#
# Valores sugeridos para probar:
#   1e-7, 5e-7, 1e-6, 5e-6, 1e-5
FACTOR_SUAVIZADO_SPLINE = 1e-6

# Ajuste polinomial:
# Para curva global: grado 4 o 5.
# Para zoom cerca del mínimo: grado 2 o 3.
GRADO_POLINOMIO = 5


# ============================================================
# 3. ESTILO LATEX REAL
# ============================================================
# Esto usa XeLaTeX para renderizar:
# - labels de ejes
# - ticks
# - leyendas
# - texto matemático
#
# Requiere tener XeLaTeX instalado en tu sistema.

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

    # Tamaño de figura.
    "figure.figsize": (5, 5 * 0.66),
    "figure.dpi": 150,
    "savefig.dpi": 400,

    # Tipografía.
    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,

    # Ejes.
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # Cuadrícula.
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.55,

    # Leyenda.
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.75",

    # Guardado.
    "savefig.bbox": "tight",
})


# ============================================================
# 4. PALETA SOBRIA
# ============================================================
# Colores pensados para tesis, impresión y buena lectura.

COLOR_ERROR_PTS = "#3296B4"    # azul sobrio
COLOR_ERROR_FIT = "#B32611"    # rojo ladrillo


# ============================================================
# 5. FUNCIONES
# ============================================================

def leer_datos() -> pd.DataFrame:
    ruta = CARPETA_DATOS / ARCHIVO

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]

    columnas_necesarias = [
        COL_R,
        COL_VQE,
        COL_FCI,
    ]

    for col in columnas_necesarias:
        if col not in df.columns:
            raise KeyError(
                f"No existe la columna '{col}' en el archivo {ARCHIVO}.\n"
                f"Columnas disponibles:\n{df.columns.tolist()}"
            )

    df = df[columnas_necesarias].dropna()
    df = df.sort_values(COL_R)

    print("\nArchivo leído correctamente:")
    print(ruta)

    print("\nColumnas disponibles usadas:")
    print(df.columns.tolist())

    print("\nPrimeras filas usadas:")
    print(df.head().to_string(index=False))

    return df


def obtener_arrays(df: pd.DataFrame):
    R = df[COL_R].to_numpy(dtype=float)
    E_vqe = df[COL_VQE].to_numpy(dtype=float)
    E_fci = df[COL_FCI].to_numpy(dtype=float)

    error_vqe_fci = np.abs(E_vqe - E_fci)

    return R, E_vqe, E_fci, error_vqe_fci


def ajuste_error(R: np.ndarray, error: np.ndarray):
    """
    Construye una curva suave para el error absoluto |E_VQE - E_FCI|.

    No es interpolación exacta. La curva puede separarse de los puntos,
    lo cual es deseable cuando hay fluctuaciones estadísticas por shots finitos.
    """

    if METODO_AJUSTE_ERROR == "ninguno":
        return None, None

    R_dense = np.linspace(R.min(), R.max(), 700)

    if METODO_AJUSTE_ERROR == "smoothing_spline":
        # Normalización para estabilidad numérica.
        R_min = R.min()
        R_scale = R.max() - R.min()

        Rn = (R - R_min) / R_scale
        Rdn = (R_dense - R_min) / R_scale

        # El parámetro s controla qué tanto se permite separar de los puntos.
        # Multiplicar por len(R) hace el parámetro menos dependiente del número de puntos.
        s = FACTOR_SUAVIZADO_SPLINE * len(R)

        spline = UnivariateSpline(Rn, error, k=3, s=s)
        error_dense = spline(Rdn)

        return R_dense, error_dense

    if METODO_AJUSTE_ERROR == "polinomio":
        coef = np.polyfit(R, error, deg=GRADO_POLINOMIO)
        polinomio = np.poly1d(coef)
        error_dense = polinomio(R_dense)

        return R_dense, error_dense

    raise ValueError(
        "METODO_AJUSTE_ERROR debe ser "
        "'smoothing_spline', 'polinomio' o 'ninguno'."
    )


def graficar():
    df = leer_datos()
    R, E_vqe, E_fci, error_vqe_fci = obtener_arrays(df)

    print("\nResumen del error absoluto:")
    print(f"Error mínimo: {error_vqe_fci.min():.12e} Eh")
    print(f"Error máximo: {error_vqe_fci.max():.12e} Eh")
    print(f"Error medio:  {error_vqe_fci.mean():.12e} Eh")

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Error: ajuste suave
    # --------------------------------------------------------
    '''
    R_fit, error_fit = ajuste_error(R, error_vqe_fci)

    if R_fit is not None:
        ax.plot(
            R_fit,
            error_fit,
            color=COLOR_ERROR_FIT,
            alpha=1,
            linestyle="-",
            linewidth=1.5,
            label=r"Ajuste suave",
            zorder=7,
        )
    '''
    # --------------------------------------------------------
    # Error: datos estimados
    # --------------------------------------------------------

    ax.scatter(
        R,
        error_vqe_fci,
        s=17,
        marker="o",
        color=COLOR_ERROR_PTS,
        alpha=1,
        linewidths=0.1,
        label=r"$|E_{\mathrm{VQE}}-E_{\mathrm{FCI}}|$",
        zorder=6,
    )

    # --------------------------------------------------------
    # Etiquetas
    # --------------------------------------------------------

    ax.set_xlabel(r"$R_{AB}$ (Å)")
    ax.set_ylabel(
        r"$|E_{\mathrm{VQE}}-E_{\mathrm{FCI}}|\,(\mathrm{E}_\mathrm{h})$"
    )

    if XLIM is not None:
        ax.set_xlim(*XLIM)

    if YLIM is not None:
        ax.set_ylim(*YLIM)
    else:
        # Como es un error absoluto, conviene iniciar el eje vertical en cero.
        ax.set_ylim(bottom=0)

    # Limpieza visual.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="best")

    fig.tight_layout()

    # --------------------------------------------------------
    # Guardado
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

    plt.show()


# ============================================================
# 6. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    graficar()