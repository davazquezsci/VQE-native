# ============================================================
# graficar_ajuste_parabolico_minimo.py
# Gráfica VQE para tesis con ajuste parabólico del mínimo.
#
# El ajuste parabólico se calcula usando los n puntos de menor
# energía estimada, pero la parábola se dibuja en todo el intervalo
# disponible de R_AB.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

# IMPORTANTE:
# El backend PGF debe activarse ANTES de importar pyplot.
mpl.use("pgf")

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_DATOS = Path("datos selectos")
CARPETA_FIGURAS = Path("figuras")
CARPETA_MIN = Path("min")

CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
CARPETA_MIN.mkdir(parents=True, exist_ok=True)

ARCHIVO = "exp03_zoom_ideal_100000.csv"
NOMBRE_FIGURA = "fig_ajuste_parabolico_zoom_ideal_100000"
NOMBRE_RESULTADOS = "ajuste_parabolico_zoom_ideal_100000.csv"

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
# 2. CONFIGURACIÓN DEL AJUSTE PARABÓLICO
# ============================================================

# Número de puntos de menor energía usados para calcular la parábola.
# Ejemplos razonables en el zoom: 15, 20, 25, 30.
N_PUNTOS_AJUSTE = 10

# Si True, marca en la gráfica los puntos usados para el ajuste.
MOSTRAR_PUNTOS_AJUSTE = True

# Si True, marca el mínimo estimado por la parábola.
MOSTRAR_MINIMO_AJUSTE = True

# Resolución de la curva parabólica dibujada en todo el intervalo.
N_PUNTOS_CURVA = 700


# ============================================================
# 3. ESTILO LATEX REAL
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

    "figure.figsize": (5, 5 * 0.66),
    "figure.dpi": 150,
    "savefig.dpi": 400,

    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.4,
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
# 4. PALETA SOBRIA
# ============================================================

COLOR_FCI = "#333333"
COLOR_HF = "#A8A8A8"
COLOR_QISKIT = "#A10787"
COLOR_VQE_PTS = "#3296B4"
COLOR_PARABOLA = "#B32611"
COLOR_PTS_AJUSTE = "#E09F3E"
COLOR_MINIMO = "#111111"


# ============================================================
# 5. FUNCIONES
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


def ajustar_parabola_minimo(df: pd.DataFrame):
    """
    Ajusta E(R) = a R^2 + b R + c usando los N_PUNTOS_AJUSTE
    datos de menor energía VQE.

    La selección se hace por energía, no por cercanía en R. Después,
    los puntos seleccionados se ordenan por R únicamente para imprimirlos
    y graficarlos de forma más clara.
    """

    if N_PUNTOS_AJUSTE < 3:
        raise ValueError("N_PUNTOS_AJUSTE debe ser al menos 3 para ajustar una parábola.")

    if N_PUNTOS_AJUSTE > len(df):
        raise ValueError(
            f"N_PUNTOS_AJUSTE={N_PUNTOS_AJUSTE} es mayor que el número de datos "
            f"disponibles: {len(df)}."
        )

    R = df[COL_R].to_numpy(dtype=float)
    E = df[COL_VQE].to_numpy(dtype=float)

    idx_menores = np.argsort(E)[:N_PUNTOS_AJUSTE]
    df_ajuste = df.iloc[idx_menores].copy()
    df_ajuste = df_ajuste.sort_values(COL_R).reset_index(drop=True)

    R_ajuste = df_ajuste[COL_R].to_numpy(dtype=float)
    E_ajuste = df_ajuste[COL_VQE].to_numpy(dtype=float)

    coef = np.polyfit(R_ajuste, E_ajuste, deg=2)
    a, b, c = coef

    if np.isclose(a, 0.0):
        raise ValueError("El coeficiente cuadrático a es prácticamente cero; no hay mínimo parabólico bien definido.")

    R_min_ajuste = -b / (2 * a)
    E_min_ajuste = a * R_min_ajuste**2 + b * R_min_ajuste + c

    idx_min_discreto = np.argmin(E)
    R_min_discreto = R[idx_min_discreto]
    E_min_discreto = E[idx_min_discreto]

    R_dense = np.linspace(R.min(), R.max(), N_PUNTOS_CURVA)
    E_dense = a * R_dense**2 + b * R_dense + c

    resultados = {
        "a": a,
        "b": b,
        "c": c,
        "R_min_discreto": R_min_discreto,
        "E_min_discreto": E_min_discreto,
        "R_min_ajuste": R_min_ajuste,
        "E_min_ajuste": E_min_ajuste,
        "ajuste_convexo": bool(a > 0),
    }

    print("\n=================================================")
    print("AJUSTE PARABÓLICO DEL MÍNIMO")
    print("=================================================")
    print(f"Archivo leído              : {CARPETA_DATOS / ARCHIVO}")
    print(f"Columna de distancia       : {COL_R}")
    print(f"Columna de energía         : {COL_VQE}")
    print(f"Puntos usados en el ajuste : {N_PUNTOS_AJUSTE}")

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


def guardar_resultados(resultados: dict):
    if not GUARDAR_RESULTADOS:
        return

    ruta_salida = CARPETA_MIN / NOMBRE_RESULTADOS

    df_resultado = pd.DataFrame({
        "archivo_origen": [str(CARPETA_DATOS / ARCHIVO)],
        "columna_distancia": [COL_R],
        "columna_energia": [COL_VQE],
        "n_puntos_ajuste": [N_PUNTOS_AJUSTE],
        "a": [resultados["a"]],
        "b": [resultados["b"]],
        "c": [resultados["c"]],
        "R_min_discreto": [resultados["R_min_discreto"]],
        "E_min_discreto": [resultados["E_min_discreto"]],
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

    R_parabola, E_parabola, df_ajuste, resultados = ajustar_parabola_minimo(df)

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Ajuste parabólico: calculado cerca del mínimo,
    # pero dibujado en todo el intervalo.
    # --------------------------------------------------------

    ax.plot(
        R_parabola,
        E_parabola,
        color=COLOR_PARABOLA,
        alpha=1,
        linestyle="-",
        linewidth=1.55,
        label=rf"VQE ideal, ajuste parabólico ($n={N_PUNTOS_AJUSTE}$)",
        zorder=8,
    )

    # --------------------------------------------------------
    # VQE: todos los datos estimados
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
        zorder=6,
    )

    # --------------------------------------------------------
    # Puntos usados para el ajuste
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
            label=r"Puntos usados en el ajuste",
            zorder=9,
        )

    # --------------------------------------------------------
    # Mínimo estimado por la parábola
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
            label=r"Mínimo del ajuste parabólico",
            zorder=10,
        )

    # --------------------------------------------------------
    # Referencias
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
    # Etiquetas
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

    guardar_resultados(resultados)

    plt.show()


# ============================================================
# 6. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    graficar()
