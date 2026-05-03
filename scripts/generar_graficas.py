# ============================================================
# generar_grafica_tesis.py
# Programa flexible para graficar resultados VQE de tesis
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Interpoladores opcionales
try:
    from scipy.interpolate import PchipInterpolator, CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_DATOS = Path("datos selectos")
CARPETA_FIGURAS = Path("figuras")
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

NOMBRE_FIGURA = "figura_vqe_exp01_global_ideal"

GUARDAR_PDF = True
GUARDAR_PNG = True


# ============================================================
# 2. ARCHIVOS A USAR
# ============================================================
# Aquí cambias los nombres según la gráfica que quieras hacer.
#
# Ejemplos:
#
# Gráfica global ideal:
#   "ideal": "exp01_global_ideal.csv"
#
# Efecto de shots:
#   "shots_100000": "exp03_zoom_ideal_100000.csv"
#   "shots_50000":  "exp04_zoom_ideal_50000.csv"
#   "shots_30000":  "exp05_zoom_ideal_30000.csv"
#
# Hardware:
#   "referencia": "exp03_zoom_ideal_100000.csv"
#   "hardware":   "5_puntos_ENERGIA_HARDWARE.csv"
#   "sim5":       "5_puntos_ENERGIA_SIMULADOS.csv"

ARCHIVOS = {
    "ideal": "exp01_global_ideal.csv",
    # "ruidoso": "exp02_global_ruidoso.csv",
    # "hardware": "5_puntos_ENERGIA_HARDWARE.csv",
}


# ============================================================
# 3. COLUMNAS POSIBLES
# ============================================================
# El programa busca automáticamente entre estos nombres.
# Puedes añadir o quitar nombres según tus CSV.

COLUMNAS = {
    "R": [
        "R_AB",
        "R",
        "distancia",
        "distancia_atomica",
        "Distancia",
    ],

    "VQE": [
        "E_VQE",
        "energia_total_vqe",
        "energia_vqe",
        "E_total_vqe",
        "energy_vqe",
    ],

    "FCI": [
        "E_FCI",
        "energia_exacta_fci",
        "energia_fci",
        "E_exacta",
        "energia_exacta",
    ],

    "HF": [
        "E_HF",
        "energia_hf",
        "energia_total_referencia",
        "reference_energy",
        "E_Hartree_Fock",
    ],

    "QISKIT": [
        "E_QISKIT",
        "energia_total_qiskit",
        "energia_qiskit",
        "E_qiskit_tot",
    ],

    "HARDWARE": [
        "E_HARDWARE",
        "energia_total_hardware",
        "energia_hardware",
        "energia_total_vqe",
        "E_VQE",
    ],
}


# ============================================================
# 4. CONFIGURACIÓN DE LA GRÁFICA
# ============================================================
# Aquí defines qué quieres mostrar.

CONFIG_GRAFICA = {
    # Título interno. Puedes dejarlo en None para tesis.
    "titulo": None,

    # Límites de ejes. Usa None si quieres automático.
    "xlim": (0.6, 1.6),
    "ylim": None,

    # Etiquetas.
    "xlabel": r"$R_{AB}\,(\mathrm{\AA})$",
    "ylabel": r"$E\,(\mathrm{Ha})$",

    # Método de suavizado para datos VQE con ruido estadístico:
    #
    # "pchip"          -> recomendado como guía visual conservadora
    # "spline"         -> spline cúbico; puede oscilar
    # "promedio_movil" -> suavizado simple por ventana
    # "ninguno"        -> sólo puntos
    "suavizado_vqe": "pchip",

    # Ventana para promedio móvil, si se usa.
    "ventana_promedio_movil": 7,

    # Mostrar puntos originales VQE.
    "mostrar_puntos_vqe": True,

    # Mostrar curva guía VQE.
    "mostrar_guia_vqe": True,
}


# ============================================================
# 5. ESTILO GENERAL DE MATPLOTLIB
# ============================================================

plt.rcParams.update({
    "figure.figsize": (6.4, 4.2),
    "figure.dpi": 150,
    "savefig.dpi": 300,

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "axes.grid": True,
    "grid.alpha": 0.30,

    "lines.linewidth": 1.6,
})


# ============================================================
# 6. FUNCIONES AUXILIARES
# ============================================================

def leer_csv(nombre_archivo):
    ruta = CARPETA_DATOS / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    return df


def encontrar_columna(df, posibles_nombres):
    """
    Busca una columna usando una lista de nombres posibles.
    No distingue mayúsculas/minúsculas.
    """
    columnas_minusculas = {c.lower(): c for c in df.columns}

    for nombre in posibles_nombres:
        if nombre.lower() in columnas_minusculas:
            return columnas_minusculas[nombre.lower()]

    raise KeyError(
        "No se encontró ninguna de estas columnas:\n"
        f"{posibles_nombres}\n\n"
        "Columnas disponibles:\n"
        f"{list(df.columns)}"
    )


def obtener_xy(df, tipo_y):
    """
    Extrae x = R_AB y y = energía del tipo indicado.

    tipo_y puede ser:
    'VQE', 'FCI', 'HF', 'QISKIT', 'HARDWARE'
    """
    col_x = encontrar_columna(df, COLUMNAS["R"])
    col_y = encontrar_columna(df, COLUMNAS[tipo_y])

    datos = df[[col_x, col_y]].dropna()
    datos = datos.sort_values(col_x)

    x = datos[col_x].to_numpy(dtype=float)
    y = datos[col_y].to_numpy(dtype=float)

    return x, y


def promedio_movil(y, ventana=7):
    """
    Suavizado simple por promedio móvil.
    Si la ventana es par, se aumenta en 1.
    """
    if ventana < 3:
        return y

    if ventana % 2 == 0:
        ventana += 1

    kernel = np.ones(ventana) / ventana
    y_suave = np.convolve(y, kernel, mode="same")

    # Para no deformar demasiado los bordes, conservamos los valores originales.
    mitad = ventana // 2
    y_suave[:mitad] = y[:mitad]
    y_suave[-mitad:] = y[-mitad:]

    return y_suave


def construir_curva_suave(x, y, metodo="pchip", ventana=7, n_puntos=500):
    """
    Devuelve una curva suave para guiar la vista.
    No debe interpretarse automáticamente como ajuste físico.
    """
    if metodo == "ninguno" or metodo is None:
        return None, None

    if len(x) < 4:
        return None, None

    x_denso = np.linspace(np.min(x), np.max(x), n_puntos)

    if metodo == "promedio_movil":
        y_suave = promedio_movil(y, ventana=ventana)

        # Interpolación lineal sólo para dibujar la línea suave.
        y_denso = np.interp(x_denso, x, y_suave)
        return x_denso, y_denso

    if not SCIPY_AVAILABLE:
        print("SciPy no está instalado. No se puede usar PCHIP ni spline.")
        return None, None

    if metodo == "pchip":
        interpolador = PchipInterpolator(x, y)
        y_denso = interpolador(x_denso)
        return x_denso, y_denso

    if metodo == "spline":
        interpolador = CubicSpline(x, y)
        y_denso = interpolador(x_denso)
        return x_denso, y_denso

    raise ValueError(
        "Método de suavizado no reconocido. Usa: "
        "'pchip', 'spline', 'promedio_movil' o 'ninguno'."
    )


def graficar_linea(ax, df, tipo_y, etiqueta, estilo="-", marcador=None):
    x, y = obtener_xy(df, tipo_y)

    ax.plot(
        x,
        y,
        linestyle=estilo,
        marker=marcador,
        markersize=3,
        label=etiqueta,
    )


def graficar_puntos(ax, df, tipo_y, etiqueta, marcador="o", tam=5):
    x, y = obtener_xy(df, tipo_y)

    ax.plot(
        x,
        y,
        linestyle="none",
        marker=marcador,
        markersize=tam,
        label=etiqueta,
    )


def graficar_vqe_estimado(
    ax,
    df,
    etiqueta_puntos="VQE, datos estimados",
    etiqueta_guia="VQE, guía visual",
    marcador="o",
    estilo_guia="-",
):
    """
    Grafica VQE como puntos + curva guía.

    Esto es lo recomendable cuando hay shots finitos y la energía sube/baja
    por ruido estadístico.
    """
    x, y = obtener_xy(df, "VQE")

    if CONFIG_GRAFICA["mostrar_puntos_vqe"]:
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=marcador,
            markersize=3.2,
            label=etiqueta_puntos,
        )

    if CONFIG_GRAFICA["mostrar_guia_vqe"]:
        x_suave, y_suave = construir_curva_suave(
            x,
            y,
            metodo=CONFIG_GRAFICA["suavizado_vqe"],
            ventana=CONFIG_GRAFICA["ventana_promedio_movil"],
        )

        if x_suave is not None:
            ax.plot(
                x_suave,
                y_suave,
                linestyle=estilo_guia,
                linewidth=1.4,
                label=etiqueta_guia,
            )


def finalizar_figura(fig, ax):
    ax.set_xlabel(CONFIG_GRAFICA["xlabel"])
    ax.set_ylabel(CONFIG_GRAFICA["ylabel"])

    if CONFIG_GRAFICA["titulo"] is not None:
        ax.set_title(CONFIG_GRAFICA["titulo"])

    if CONFIG_GRAFICA["xlim"] is not None:
        ax.set_xlim(*CONFIG_GRAFICA["xlim"])

    if CONFIG_GRAFICA["ylim"] is not None:
        ax.set_ylim(*CONFIG_GRAFICA["ylim"])

    ax.legend(frameon=True)
    fig.tight_layout()

    if GUARDAR_PDF:
        ruta_pdf = CARPETA_FIGURAS / f"{NOMBRE_FIGURA}.pdf"
        fig.savefig(ruta_pdf, bbox_inches="tight")
        print(f"Guardado: {ruta_pdf}")

    if GUARDAR_PNG:
        ruta_png = CARPETA_FIGURAS / f"{NOMBRE_FIGURA}.png"
        fig.savefig(ruta_png, bbox_inches="tight")
        print(f"Guardado: {ruta_png}")

    plt.show()


# ============================================================
# 7. ZONA QUE MODIFICAS PARA CADA GRÁFICA
# ============================================================

def main():
    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Carga de archivos
    # --------------------------------------------------------
    datos = {}

    for clave, archivo in ARCHIVOS.items():
        datos[clave] = leer_csv(archivo)

    # ========================================================
    # EJEMPLO A: Curva global ideal
    # Exp. 1: [0.6, 1.6] Å, 100000 shots, maxiter=100
    # ========================================================
    #
    # ARCHIVOS = {
    #     "ideal": "exp01_global_ideal.csv",
    # }
    #
    # CONFIG_GRAFICA["xlim"] = (0.6, 1.6)
    # NOMBRE_FIGURA = "fig01_curva_global_ideal"
    #
    # Código:
    #
    graficar_linea(
        ax,
        datos["ideal"],
        "FCI",
        etiqueta="FCI",
        estilo="-",
    )

    graficar_linea(
        ax,
        datos["ideal"],
        "HF",
        etiqueta="Hartree--Fock",
        estilo="--",
    )

    # Qiskit es opcional. Si tu archivo no tiene columna Qiskit,
    # comenta este bloque.
    try:
        graficar_linea(
            ax,
            datos["ideal"],
            "QISKIT",
            etiqueta="Qiskit",
            estilo="-.",
        )
    except KeyError:
        pass

    graficar_vqe_estimado(
        ax,
        datos["ideal"],
        etiqueta_puntos="VQE ideal, datos estimados",
        etiqueta_guia="VQE ideal, guía visual",
        marcador="o",
        estilo_guia="-",
    )

    # ========================================================
    # EJEMPLO B: Efecto de shots
    # ========================================================
    #
    # Para usar este ejemplo, cambia ARCHIVOS arriba:
    #
    # ARCHIVOS = {
    #     "s100": "exp03_zoom_ideal_100000.csv",
    #     "s50":  "exp04_zoom_ideal_50000.csv",
    #     "s30":  "exp05_zoom_ideal_30000.csv",
    # }
    #
    # Y comenta el EJEMPLO A.
    #
    # Luego usa:
    #
    # graficar_linea(ax, datos["s100"], "FCI", "FCI", "-")
    #
    # CONFIG_GRAFICA["suavizado_vqe"] = "pchip"
    #
    # x, y = obtener_xy(datos["s100"], "VQE")
    # ax.plot(x, y, "o", markersize=3, label="VQE, 100000 shots")
    # xs, ys = construir_curva_suave(x, y, metodo="pchip")
    # ax.plot(xs, ys, "-", linewidth=1.3, label="Guía, 100000 shots")
    #
    # x, y = obtener_xy(datos["s50"], "VQE")
    # ax.plot(x, y, "s", markersize=3, label="VQE, 50000 shots")
    # xs, ys = construir_curva_suave(x, y, metodo="pchip")
    # ax.plot(xs, ys, "--", linewidth=1.3, label="Guía, 50000 shots")
    #
    # x, y = obtener_xy(datos["s30"], "VQE")
    # ax.plot(x, y, "^", markersize=3, label="VQE, 30000 shots")
    # xs, ys = construir_curva_suave(x, y, metodo="pchip")
    # ax.plot(xs, ys, ":", linewidth=1.3, label="Guía, 30000 shots")

    # ========================================================
    # EJEMPLO C: Hardware de 5 puntos
    # ========================================================
    #
    # Para usar este ejemplo, cambia ARCHIVOS arriba:
    #
    # ARCHIVOS = {
    #     "ref": "exp03_zoom_ideal_100000.csv",
    #     "hardware": "5_puntos_ENERGIA_HARDWARE.csv",
    #     "sim5": "5_puntos_ENERGIA_SIMULADOS.csv",
    # }
    #
    # Y comenta el EJEMPLO A.
    #
    # Luego usa:
    #
    # graficar_linea(ax, datos["ref"], "FCI", "FCI", "-")
    # graficar_linea(ax, datos["ref"], "HF", "Hartree--Fock", "--")
    # graficar_puntos(ax, datos["sim5"], "VQE", "VQE simulado, 5 puntos", "s", 5)
    # graficar_puntos(ax, datos["hardware"], "HARDWARE", "Hardware real", "o", 6)
    #
    # IMPORTANTE:
    # Hardware real va como puntos, no como curva continua.

    finalizar_figura(fig, ax)


if __name__ == "__main__":
    main()