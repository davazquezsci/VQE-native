import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# CONFIGURACIÓN
# =========================================================

archivo_csv = "datos selectos/exp03_zoom_ideal_100000.csv"

# Columnas a usar
columna_distancia = "R_AB"
columna_energia = "E_VQE_sim_id"

# Número de puntos de menor energía usados para el ajuste
n_puntos_ajuste = 25

# Archivo de salida
archivo_salida = "min/ajuste_minimo_E_VQE_sim_id_zoom_ideal_100000.csv"


# =========================================================
# CARGA DE DATOS
# =========================================================

df = pd.read_csv(archivo_csv)

# Ordenar por seguridad respecto a la distancia internuclear
df = df.sort_values(columna_distancia).reset_index(drop=True)

# Verificar que existan las columnas solicitadas
columnas_necesarias = {columna_distancia, columna_energia}
columnas_faltantes = columnas_necesarias - set(df.columns)

if columnas_faltantes:
    raise ValueError(
        f"Faltan columnas en el archivo CSV: {columnas_faltantes}. "
        f"Columnas disponibles: {list(df.columns)}"
    )

R = df[columna_distancia].to_numpy()
E = df[columna_energia].to_numpy()


# =========================================================
# SELECCIONAR LOS 7 DATOS DE MENOR ENERGÍA
# =========================================================

if n_puntos_ajuste > len(df):
    raise ValueError(
        f"n_puntos_ajuste={n_puntos_ajuste} es mayor que el número de datos "
        f"disponibles: {len(df)}."
    )

# Índices de los n puntos con menor energía
idx_menores = np.argsort(E)[:n_puntos_ajuste]

df_ajuste = df.iloc[idx_menores].copy()

# Ordenar los puntos seleccionados por distancia antes del ajuste
df_ajuste = df_ajuste.sort_values(columna_distancia).reset_index(drop=True)

R_ajuste = df_ajuste[columna_distancia].to_numpy()
E_ajuste = df_ajuste[columna_energia].to_numpy()

print("=================================================")
print("AJUSTE PARABÓLICO DEL MÍNIMO")
print("=================================================")
print(f"Archivo leído              : {archivo_csv}")
print(f"Columna de distancia       : {columna_distancia}")
print(f"Columna de energía         : {columna_energia}")
print(f"Puntos usados en el ajuste : {n_puntos_ajuste}")

print("\nPuntos seleccionados para el ajuste:")
print(df_ajuste[[columna_distancia, columna_energia]])


# =========================================================
# AJUSTE PARABÓLICO: E(R) = a R^2 + b R + c
# =========================================================

coef = np.polyfit(R_ajuste, E_ajuste, deg=2)
a, b, c = coef

if a <= 0:
    print("\nADVERTENCIA: el ajuste no es convexo (a <= 0).")
    print("El mínimo parabólico puede no ser físicamente representativo.")

R_min_ajuste = -b / (2 * a)
E_min_ajuste = a * R_min_ajuste**2 + b * R_min_ajuste + c

# Mínimo discreto dentro de todos los datos
idx_min_discreto = np.argmin(E)
R_min_discreto = R[idx_min_discreto]
E_min_discreto = E[idx_min_discreto]

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


# =========================================================
# GUARDAR RESULTADOS
# =========================================================

df_resultado = pd.DataFrame({
    "archivo_origen": [archivo_csv],
    "columna_distancia": [columna_distancia],
    "columna_energia": [columna_energia],
    "n_puntos_ajuste": [n_puntos_ajuste],
    "a": [a],
    "b": [b],
    "c": [c],
    "R_min_discreto": [R_min_discreto],
    "E_min_discreto": [E_min_discreto],
    "R_min_ajuste": [R_min_ajuste],
    "E_min_ajuste": [E_min_ajuste],
    "ajuste_convexo": [a > 0],
})

Path(archivo_salida).parent.mkdir(parents=True, exist_ok=True)
df_resultado.to_csv(archivo_salida, index=False)

print("\nArchivo de resultados generado:")
print(archivo_salida)