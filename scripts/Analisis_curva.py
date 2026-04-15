import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


# =========================================================
# CONFIGURACIÓN
# =========================================================

archivo_csv = "datos/HeHplus_40 pts zoom 50000 v2.csv"

# Curva usada para localizar el mínimo:
# opciones: "E_exacta", "E_qiskit_tot", "E_VQE", "E_HF"
columna_referencia = "E_exacta"

# Número de puntos vecinos al mínimo para el ajuste parabólico local
# Por ejemplo 7 usa el punto mínimo y 3 vecinos a cada lado
n_puntos_ajuste = 7

# Separación para proponer 5 puntos alrededor del mínimo estimado
delta_R = 0.01  # en Angstrom


# =========================================================
# CARGA DE DATOS
# =========================================================

df = pd.read_csv(archivo_csv)

# Ordenar por seguridad
df = df.sort_values("R_AB").reset_index(drop=True)

R = df["R_AB"].to_numpy()
E_ref = df[columna_referencia].to_numpy()

# =========================================================
# LOCALIZAR MÍNIMO DISCRETO
# =========================================================

idx_min = np.argmin(E_ref)
R_min_discreto = R[idx_min]
E_min_discreto = E_ref[idx_min]

print("=================================================")
print("ANÁLISIS DEL MÍNIMO LOCAL")
print("=================================================")
print(f"Curva de referencia usada   : {columna_referencia}")
print(f"Mínimo discreto encontrado  :")
print(f"   R = {R_min_discreto:.6f} Å")
print(f"   E = {E_min_discreto:.12f} Hartree")

# =========================================================
# SELECCIONAR VENTANA LOCAL
# =========================================================

half_window = n_puntos_ajuste // 2
i0 = max(0, idx_min - half_window)
i1 = min(len(df), idx_min + half_window + 1)

df_local = df.iloc[i0:i1].copy()

R_local = df_local["R_AB"].to_numpy()
E_local = df_local[columna_referencia].to_numpy()

print("\nPuntos usados en el ajuste local:")
print(df_local[["R_AB", columna_referencia]])

# =========================================================
# AJUSTE PARABÓLICO: E(R) = a R^2 + b R + c
# =========================================================

coef = np.polyfit(R_local, E_local, deg=2)
a, b, c = coef

print("\nCoeficientes del ajuste parabólico:")
print(f"a = {a:.12f}")
print(f"b = {b:.12f}")
print(f"c = {c:.12f}")

if a <= 0:
    print("\nADVERTENCIA: el ajuste no es convexo (a <= 0).")
    print("Revisa la ventana de puntos usada para el ajuste.")

R_min_ajuste = -b / (2 * a)
E_min_ajuste = a * R_min_ajuste**2 + b * R_min_ajuste + c

print("\nMínimo estimado por ajuste local:")
print(f"   R* = {R_min_ajuste:.6f} Å")
print(f"   E* = {E_min_ajuste:.12f} Hartree")

# =========================================================
# PROPUESTA DE 5 PUNTOS PARA HARDWARE
# =========================================================

puntos_hardware = np.array([
    R_min_ajuste - 2 * delta_R,
    R_min_ajuste - 1 * delta_R,
    R_min_ajuste,
    R_min_ajuste + 1 * delta_R,
    R_min_ajuste + 2 * delta_R,
])

print("\nPropuesta de 5 puntos centrados en el mínimo ajustado:")
for r in puntos_hardware:
    print(f"   {r:.6f} Å")

# =========================================================
# GUARDAR PROPUESTA EN CSV
# =========================================================

df_puntos = pd.DataFrame({
    "R_AB_sugerido": puntos_hardware
})
df_puntos.to_csv("datos/5_puntos_hardware_sugeridos.csv", index=False)

# =========================================================
# GRAFICAR
# =========================================================

# Malla fina para dibujar la parábola ajustada
R_fit = np.linspace(R_local.min(), R_local.max(), 300)
E_fit = a * R_fit**2 + b * R_fit + c

plt.figure(figsize=(7, 5))

# Curvas disponibles
plt.plot(df["R_AB"], df["E_exacta"], "o-", label="Exacta")
plt.plot(df["R_AB"], df["E_VQE"], "o-", label="VQE")
plt.plot(df["R_AB"], df["E_HF"], "o-", label="Hartree-Fock")

# Ajuste local
plt.plot(R_fit, E_fit, "--", label=f"Ajuste parabólico ({columna_referencia})")

# Mínimo discreto y ajustado
plt.axvline(R_min_discreto, linestyle=":", label="Mínimo discreto")
plt.axvline(R_min_ajuste, linestyle="-.", label="Mínimo ajustado")

# Puntos sugeridos
for r in puntos_hardware:
    plt.axvline(r, alpha=0.35)

plt.xlabel(r"Distancia internuclear $R_{AB}$ ($\AA$)")
plt.ylabel("Energía total (Hartree)")
plt.title("Ajuste local del mínimo energético")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graficas/ajuste_local_minimo.png", dpi=300)
plt.show()