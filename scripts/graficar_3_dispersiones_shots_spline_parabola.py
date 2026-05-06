# ============================================================
# graficar_3_dispersiones_shots_spline_parabola.py
#
# Gráfica comparativa para tesis:
#   1) curva FCI de referencia,
#   2) tres dispersiones VQE ideal con diferente número de shots,
#   3) spline suavizado para cada curva VQE,
#   4) ajuste parabólico local para cada curva VQE.
#
# Además genera un archivo CSV con los mínimos estimados para cada
# curva VQE y sus errores respecto a FCI.
#
# Importante:
#   - NO se grafica la región usada para el ajuste parabólico.
#   - La parábola se calcula localmente cerca del mínimo del spline,
#     pero se dibuja en todo el intervalo de R_AB.
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

# Si tus archivos están dentro de "datos selectos", deja esta ruta.
# Si están en la misma carpeta que el programa, el script también los buscará ahí.
CARPETA_DATOS = Path("datos selectos")
CARPETA_FIGURAS = Path("figuras")
CARPETA_MIN = Path("min")

CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
CARPETA_MIN.mkdir(parents=True, exist_ok=True)

NOMBRE_FIGURA = "fig_comparacion_shots_ideal_spline_parabola"
NOMBRE_RESULTADOS = "minimos_comparacion_shots_ideal_spline_parabola.csv"
NOMBRE_RESUMEN_TXT = "resumen_minimos_comparacion_shots_ideal_spline_parabola.txt"

# Columnas exactas de tus CSV.
COL_R = "R_AB"
COL_VQE = "E_VQE_sim_id"
COL_FCI = "E_exacta"

# Archivos a comparar.
# Puedes cambiar las etiquetas si quieres que aparezcan de otra forma en la leyenda.
CURVAS_VQE = [
    {
        "archivo": "exp03_zoom_ideal_100000.csv",
        "shots": 100000,
        "etiqueta": r"VQE ideal, $100000$ shots",
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "archivo": "exp04_zoom_ideal_50000.csv",
        "shots": 50000,
        "etiqueta": r"VQE ideal, $50000$ shots",
        "color": "#d62728",
        "marker": "s",
    },
    {
        "archivo": "exp05_zoom_ideal_30000.csv",
        "shots": 30000,
        "etiqueta": r"VQE ideal, $30000$ shots",
        "color": "#2ca02c",
        "marker": "^",
    },
]

# Límites visuales. Si quieres usar todo el intervalo del archivo, deja XLIM = None.
XLIM = (0.8, 1.1)
YLIM = None

# Guardar archivos.
GUARDAR_PDF = True
GUARDAR_PGF = True
GUARDAR_PNG = False
GUARDAR_RESULTADOS = True


# ============================================================
# 2. CONFIGURACIÓN DEL SPLINE GLOBAL
# ============================================================

MOSTRAR_SPLINE_GLOBAL = True

# Más grande = curva más holgada.
# Más chico = curva más pegada a los puntos.
# Valores sugeridos: 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3.
FACTOR_SUAVIZADO_SPLINE = 1e-6
GRADO_SPLINE = 3


# ============================================================
# 3. CONFIGURACIÓN DEL AJUSTE PARABÓLICO LOCAL
# ============================================================

# Opciones:
#   "ventana_spline" -> usa una vecindad continua alrededor del mínimo del spline.
#   "vecinos_spline" -> usa los N_PUNTOS_AJUSTE puntos más cercanos al mínimo del spline.
MODO_SELECCION_AJUSTE = "vecinos_spline"

# Si usas MODO_SELECCION_AJUSTE = "ventana_spline", se toma:
# [R_min_spline - DELTA_R_AJUSTE, R_min_spline + DELTA_R_AJUSTE].
DELTA_R_AJUSTE = 0.04

# Si usas MODO_SELECCION_AJUSTE = "vecinos_spline".
N_PUNTOS_AJUSTE = 30

# La parábola se dibuja en todo el intervalo, aunque se ajuste localmente.
MOSTRAR_PARABOLA = True

# Marcadores opcionales para los mínimos. Por defecto se dejan apagados para
# mantener la figura limpia.
MOSTRAR_MINIMOS = False

# Resolución de curvas suaves.
N_PUNTOS_CURVA = 900


# ============================================================
# 4. CONFIGURACIÓN DEL MÍNIMO FCI DE REFERENCIA
# ============================================================

# Para estimar el mínimo FCI se usa un spline interpolante de FCI.
# Este spline NO se grafica; sólo sirve como referencia numérica.
N_PUNTOS_CURVA_FCI = 3000


# ============================================================
# 5. ESTILO LATEX REAL
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

    "figure.figsize": (6.4, 6.4 * 0.66),
    "figure.dpi": 150,
    "savefig.dpi": 400,

    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 7.4,
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
# 6. PALETA Y ESTILO DE LÍNEAS
# ============================================================

COLOR_FCI = "#333333"
COLOR_MINIMO = "#111111"

ALPHA_PUNTOS = 0.72
TAMANO_PUNTOS = 15
LINEWIDTH_SPLINE = 1.45
LINEWIDTH_PARABOLA = 1.25
LINEWIDTH_FCI = 1.95


# ============================================================
# 7. FUNCIONES AUXILIARES
# ============================================================

def ruta_datos(nombre_archivo: str) -> Path:
    """
    Busca el archivo primero en CARPETA_DATOS y luego en la carpeta actual.
    Esto permite ejecutar el script tanto desde tu estructura normal de tesis
    como desde una carpeta donde estén los CSV directamente.
    """
    ruta_1 = CARPETA_DATOS / nombre_archivo
    ruta_2 = Path(nombre_archivo)

    if ruta_1.exists():
        return ruta_1
    if ruta_2.exists():
        return ruta_2

    raise FileNotFoundError(
        f"No se encontró el archivo '{nombre_archivo}'.\n"
        f"Busqué en:\n  {ruta_1}\n  {ruta_2}"
    )


def leer_csv_curva(config: dict) -> pd.DataFrame:
    ruta = ruta_datos(config["archivo"])
    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]

    columnas_necesarias = [COL_R, COL_VQE, COL_FCI]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Faltan columnas en {ruta}: {faltantes}.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    df = df[columnas_necesarias].dropna()
    df = df.sort_values(COL_R).reset_index(drop=True)

    return df


def construir_spline(R: np.ndarray, E: np.ndarray, factor_suavizado: float):
    """
    Construye un spline suavizado en coordenada normalizada.
    Devuelve el spline y los parámetros de normalización.
    """
    if len(R) <= GRADO_SPLINE:
        raise ValueError(
            f"No hay suficientes puntos para un spline de grado k={GRADO_SPLINE}."
        )

    R_min = R.min()
    R_scale = R.max() - R.min()

    if np.isclose(R_scale, 0.0):
        raise ValueError("El intervalo de R tiene longitud cero.")

    Rn = (R - R_min) / R_scale
    s = factor_suavizado * len(R)

    spline = UnivariateSpline(Rn, E, k=GRADO_SPLINE, s=s)

    return spline, R_min, R_scale


def evaluar_spline(spline, R_min: float, R_scale: float, R_eval: np.ndarray):
    Rn_eval = (R_eval - R_min) / R_scale
    return spline(Rn_eval)


def minimo_spline(R: np.ndarray, spline, R_min_norm: float, R_scale: float):
    R_dense = np.linspace(R.min(), R.max(), max(N_PUNTOS_CURVA, 2000))
    E_dense = evaluar_spline(spline, R_min_norm, R_scale, R_dense)
    idx = np.argmin(E_dense)
    return R_dense[idx], E_dense[idx]


def seleccionar_puntos_locales(df: pd.DataFrame, R_min_spline: float) -> pd.DataFrame:
    R = df[COL_R].to_numpy(dtype=float)

    if MODO_SELECCION_AJUSTE == "ventana_spline":
        if DELTA_R_AJUSTE <= 0:
            raise ValueError("DELTA_R_AJUSTE debe ser positivo.")

        mascara = (R >= R_min_spline - DELTA_R_AJUSTE) & (R <= R_min_spline + DELTA_R_AJUSTE)
        df_ajuste = df.loc[mascara].copy()

        if len(df_ajuste) < 3:
            raise ValueError(
                "La ventana alrededor del mínimo del spline contiene menos de 3 puntos. "
                "Aumenta DELTA_R_AJUSTE o cambia a 'vecinos_spline'."
            )

    elif MODO_SELECCION_AJUSTE == "vecinos_spline":
        if N_PUNTOS_AJUSTE < 3:
            raise ValueError("N_PUNTOS_AJUSTE debe ser al menos 3.")
        if N_PUNTOS_AJUSTE > len(df):
            raise ValueError(
                f"N_PUNTOS_AJUSTE={N_PUNTOS_AJUSTE} es mayor que el número de datos "
                f"disponibles: {len(df)}."
            )

        idx_vecinos = np.argsort(np.abs(R - R_min_spline))[:N_PUNTOS_AJUSTE]
        df_ajuste = df.iloc[idx_vecinos].copy()

    else:
        raise ValueError(
            "MODO_SELECCION_AJUSTE debe ser 'ventana_spline' o 'vecinos_spline'."
        )

    return df_ajuste.sort_values(COL_R).reset_index(drop=True)


def ajustar_parabola(df_ajuste: pd.DataFrame):
    R_ajuste = df_ajuste[COL_R].to_numpy(dtype=float)
    E_ajuste = df_ajuste[COL_VQE].to_numpy(dtype=float)

    coef = np.polyfit(R_ajuste, E_ajuste, deg=2)
    a, b, c = coef

    if np.isclose(a, 0.0):
        raise ValueError("El coeficiente cuadrático a es prácticamente cero.")

    R_min = -b / (2 * a)
    E_min = a * R_min**2 + b * R_min + c

    return a, b, c, R_min, E_min


def estimar_minimo_fci(df_ref: pd.DataFrame):
    """
    Estima el mínimo FCI usando un spline interpolante de la columna FCI.
    No se grafica este spline; sólo sirve como referencia numérica.
    """
    R = df_ref[COL_R].to_numpy(dtype=float)
    E_fci = df_ref[COL_FCI].to_numpy(dtype=float)

    spline_fci, R_min_norm, R_scale = construir_spline(
        R, E_fci, factor_suavizado=0.0
    )

    R_dense = np.linspace(R.min(), R.max(), N_PUNTOS_CURVA_FCI)
    E_dense = evaluar_spline(spline_fci, R_min_norm, R_scale, R_dense)
    idx = np.argmin(E_dense)

    R_min_fci = R_dense[idx]
    E_min_fci = E_dense[idx]

    return {
        "spline": spline_fci,
        "R_min_norm": R_min_norm,
        "R_scale": R_scale,
        "R_min_FCI": R_min_fci,
        "E_min_FCI": E_min_fci,
    }


def procesar_curva(config: dict, ref_fci: dict) -> dict:
    df = leer_csv_curva(config)
    R = df[COL_R].to_numpy(dtype=float)
    E_vqe = df[COL_VQE].to_numpy(dtype=float)

    spline, R_min_norm, R_scale = construir_spline(
        R, E_vqe, factor_suavizado=FACTOR_SUAVIZADO_SPLINE
    )

    R_min_spline, E_min_spline = minimo_spline(R, spline, R_min_norm, R_scale)

    df_ajuste = seleccionar_puntos_locales(df, R_min_spline)
    a, b, c, R_min_parabola, E_min_parabola = ajustar_parabola(df_ajuste)

    idx_min_discreto = np.argmin(E_vqe)
    R_min_discreto = R[idx_min_discreto]
    E_min_discreto = E_vqe[idx_min_discreto]

    R_dense = np.linspace(R.min(), R.max(), N_PUNTOS_CURVA)
    E_spline_dense = evaluar_spline(spline, R_min_norm, R_scale, R_dense)
    E_parabola_dense = a * R_dense**2 + b * R_dense + c

    E_fci_en_Rmin_vqe = evaluar_spline(
        ref_fci["spline"],
        ref_fci["R_min_norm"],
        ref_fci["R_scale"],
        np.array([R_min_parabola]),
    )[0]

    resultado = {
        "config": config,
        "df": df,
        "df_ajuste": df_ajuste,
        "R_dense": R_dense,
        "E_spline_dense": E_spline_dense,
        "E_parabola_dense": E_parabola_dense,
        "a": a,
        "b": b,
        "c": c,
        "R_min_discreto": R_min_discreto,
        "E_min_discreto": E_min_discreto,
        "R_min_spline": R_min_spline,
        "E_min_spline": E_min_spline,
        "R_min_parabola": R_min_parabola,
        "E_min_parabola": E_min_parabola,
        "R_region_ajuste_min": df_ajuste[COL_R].min(),
        "R_region_ajuste_max": df_ajuste[COL_R].max(),
        "n_puntos_usados": len(df_ajuste),
        "ajuste_convexo": bool(a > 0),
        "E_FCI_en_R_min_parabola": E_fci_en_Rmin_vqe,
        "error_R_min_vs_FCI_abs_A": abs(R_min_parabola - ref_fci["R_min_FCI"]),
        "error_E_min_vs_FCI_min_abs_Eh": abs(E_min_parabola - ref_fci["E_min_FCI"]),
        "error_E_vs_FCI_en_Rmin_abs_Eh": abs(E_min_parabola - E_fci_en_Rmin_vqe),
    }

    return resultado


def guardar_resultados(resultados: list, ref_fci: dict):
    if not GUARDAR_RESULTADOS:
        return

    filas = []
    for res in resultados:
        cfg = res["config"]
        filas.append({
            "archivo": cfg["archivo"],
            "shots": cfg["shots"],
            "modo_seleccion_ajuste": MODO_SELECCION_AJUSTE,
            "n_puntos_ajuste_configurado": N_PUNTOS_AJUSTE,
            "n_puntos_usados": res["n_puntos_usados"],
            "delta_R_ajuste": DELTA_R_AJUSTE if MODO_SELECCION_AJUSTE == "ventana_spline" else np.nan,
            "factor_suavizado_spline": FACTOR_SUAVIZADO_SPLINE,
            "grado_spline": GRADO_SPLINE,
            "a": res["a"],
            "b": res["b"],
            "c": res["c"],
            "ajuste_convexo": res["ajuste_convexo"],
            "R_region_ajuste_min_A": res["R_region_ajuste_min"],
            "R_region_ajuste_max_A": res["R_region_ajuste_max"],
            "R_min_discreto_A": res["R_min_discreto"],
            "E_min_discreto_Eh": res["E_min_discreto"],
            "R_min_spline_A": res["R_min_spline"],
            "E_min_spline_Eh": res["E_min_spline"],
            "R_min_parabola_A": res["R_min_parabola"],
            "E_min_parabola_Eh": res["E_min_parabola"],
            "R_min_FCI_A": ref_fci["R_min_FCI"],
            "E_min_FCI_Eh": ref_fci["E_min_FCI"],
            "E_FCI_en_R_min_parabola_Eh": res["E_FCI_en_R_min_parabola"],
            "error_R_min_vs_FCI_abs_A": res["error_R_min_vs_FCI_abs_A"],
            "error_E_min_vs_FCI_min_abs_Eh": res["error_E_min_vs_FCI_min_abs_Eh"],
            "error_E_vs_FCI_en_Rmin_abs_Eh": res["error_E_vs_FCI_en_Rmin_abs_Eh"],
        })

    df_res = pd.DataFrame(filas)
    ruta_csv = CARPETA_MIN / NOMBRE_RESULTADOS
    df_res.to_csv(ruta_csv, index=False)

    ruta_txt = CARPETA_MIN / NOMBRE_RESUMEN_TXT
    lineas = []
    lineas.append("RESUMEN DE MÍNIMOS PARA COMPARACIÓN DE SHOTS\n")
    lineas.append("=============================================\n\n")
    lineas.append(f"Mínimo FCI de referencia:\n")
    lineas.append(f"R_min_FCI = {ref_fci['R_min_FCI']:.12f} Å\n")
    lineas.append(f"E_min_FCI = {ref_fci['E_min_FCI']:.12f} Eh\n\n")
    lineas.append(f"Modo de selección para parábola: {MODO_SELECCION_AJUSTE}\n")
    lineas.append(f"N_PUNTOS_AJUSTE = {N_PUNTOS_AJUSTE}\n")
    lineas.append(f"DELTA_R_AJUSTE = {DELTA_R_AJUSTE}\n")
    lineas.append(f"FACTOR_SUAVIZADO_SPLINE = {FACTOR_SUAVIZADO_SPLINE}\n\n")

    for res in resultados:
        cfg = res["config"]
        lineas.append(f"{cfg['archivo']} ({cfg['shots']} shots)\n")
        lineas.append("-" * 60 + "\n")
        lineas.append(f"R_min_parabola = {res['R_min_parabola']:.12f} Å\n")
        lineas.append(f"E_min_parabola = {res['E_min_parabola']:.12f} Eh\n")
        lineas.append(f"R_min_spline = {res['R_min_spline']:.12f} Å\n")
        lineas.append(f"E_min_spline = {res['E_min_spline']:.12f} Eh\n")
        lineas.append(f"error_R_min_vs_FCI_abs = {res['error_R_min_vs_FCI_abs_A']:.12e} Å\n")
        lineas.append(f"error_E_min_vs_FCI_min_abs = {res['error_E_min_vs_FCI_min_abs_Eh']:.12e} Eh\n")
        lineas.append(f"error_E_vs_FCI_en_Rmin_abs = {res['error_E_vs_FCI_en_Rmin_abs_Eh']:.12e} Eh\n")
        lineas.append("\n")

    ruta_txt.write_text("".join(lineas), encoding="utf-8")

    print("\nArchivos de mínimos generados:")
    print(ruta_csv)
    print(ruta_txt)


def graficar():
    # Se usa el primer archivo como referencia para FCI.
    df_ref = leer_csv_curva(CURVAS_VQE[0])
    R_ref = df_ref[COL_R].to_numpy(dtype=float)
    E_fci_ref = df_ref[COL_FCI].to_numpy(dtype=float)
    ref_fci = estimar_minimo_fci(df_ref)

    resultados = [procesar_curva(cfg, ref_fci) for cfg in CURVAS_VQE]

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # FCI de referencia.
    # --------------------------------------------------------
    ax.plot(
        R_ref,
        E_fci_ref,
        color=COLOR_FCI,
        linestyle="-",
        linewidth=LINEWIDTH_FCI,
        label=r"FCI",
        zorder=5,
    )

    # --------------------------------------------------------
    # Tres curvas VQE: dispersión, spline y parábola.
    # --------------------------------------------------------
    for i, res in enumerate(resultados):
        cfg = res["config"]
        df = res["df"]
        color = cfg["color"]

        R = df[COL_R].to_numpy(dtype=float)
        E_vqe = df[COL_VQE].to_numpy(dtype=float)

        # Dispersión VQE.
        ax.scatter(
            R,
            E_vqe,
            s=TAMANO_PUNTOS,
            marker=cfg["marker"],
            color=color,
            alpha=ALPHA_PUNTOS,
            linewidths=0.1,
            label=cfg["etiqueta"],
            zorder=7 + i,
        )

        # Spline global.
        if MOSTRAR_SPLINE_GLOBAL:
            ax.plot(
                res["R_dense"],
                res["E_spline_dense"],
                color=color,
                linestyle="-",
                linewidth=LINEWIDTH_SPLINE,
                alpha=1.0,
                label=rf"Spline, {cfg['shots']} shots",
                zorder=10 + i,
            )

        # Parábola local, dibujada en todo el intervalo.
        if MOSTRAR_PARABOLA:
            ax.plot(
                res["R_dense"],
                res["E_parabola_dense"],
                color=color,
                linestyle="--",
                linewidth=LINEWIDTH_PARABOLA,
                alpha=1.0,
                label=rf"Parábola, {cfg['shots']} shots",
                zorder=13 + i,
            )

        if MOSTRAR_MINIMOS:
            ax.scatter(
                [res["R_min_parabola"]],
                [res["E_min_parabola"]],
                s=30,
                marker="D",
                color=COLOR_MINIMO,
                alpha=1,
                linewidths=0.5,
                zorder=20 + i,
            )

    # --------------------------------------------------------
    # Etiquetas y formato.
    # --------------------------------------------------------
    ax.set_xlabel(r"$R_{AB}$ (Å)")
    ax.set_ylabel(r"$E\,(\mathrm{E}_\mathrm{h})$")

    if XLIM is not None:
        ax.set_xlim(*XLIM)
    if YLIM is not None:
        ax.set_ylim(*YLIM)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="best", ncol=1)
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

    guardar_resultados(resultados, ref_fci)

    print("\n=================================================")
    print("RESUMEN DE MÍNIMOS")
    print("=================================================")
    print(f"FCI: R_min = {ref_fci['R_min_FCI']:.8f} Å, E_min = {ref_fci['E_min_FCI']:.12f} Eh")
    for res in resultados:
        cfg = res["config"]
        print(
            f"{cfg['shots']:>6} shots: "
            f"R_min(parábola) = {res['R_min_parabola']:.8f} Å, "
            f"E_min(parábola) = {res['E_min_parabola']:.12f} Eh, "
            f"|ΔE_min| = {res['error_E_min_vs_FCI_min_abs_Eh']:.6e} Eh"
        )

    plt.show()


# ============================================================
# 8. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    graficar()
