"""
Pendulo - Grupo 15
Lee los datos del Excel y calcula velocidad/aceleracion angular vs. tiempo.

Instalar (una sola vez):
    pip install pandas openpyxl numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIGURACION (ajustar a tu Excel) ----------------
ARCHIVO = "datos_pendulo.xlsx"   # nombre del Excel
COL_TIEMPO = "t"                 # nombre de la columna de tiempo (s)
COL_ANGULO = "theta"             # nombre de la columna de angulo
ANGULO_EN_GRADOS = True          # False si ya esta en radianes
# --------------------------------------------------------------------

# sheet_name=None lee TODAS las hojas -> dict {nombre_hoja: DataFrame}
# (util si cada hoja es una amplitud o longitud distinta)
hojas = pd.read_excel(ARCHIVO, sheet_name=None)

print("Hojas encontradas:", list(hojas.keys()))
for nombre, df in hojas.items():
    print(f"\n--- {nombre} ---")
    print("Columnas:", list(df.columns))
    print(df.head())


def analizar(df):
    """Devuelve un DataFrame con theta, omega y alfa (rad, rad/s, rad/s^2)."""
    df = df[[COL_TIEMPO, COL_ANGULO]].dropna().sort_values(COL_TIEMPO)
    t = df[COL_TIEMPO].to_numpy(dtype=float)
    theta = df[COL_ANGULO].to_numpy(dtype=float)
    if ANGULO_EN_GRADOS:
        theta = np.deg2rad(theta)

    # Derivadas numericas (diferencias centradas)
    omega = np.gradient(theta, t)   # velocidad angular
    alfa = np.gradient(omega, t)    # aceleracion angular
    return pd.DataFrame({"t": t, "theta": theta, "omega": omega, "alfa": alfa})


resultados = {}
for nombre, df in hojas.items():
    try:
        resultados[nombre] = analizar(df)
    except KeyError:
        print(f"[!] La hoja '{nombre}' no tiene las columnas "
              f"'{COL_TIEMPO}' y '{COL_ANGULO}'. Revisar la configuracion.")

# Grafico rapido: angulo vs tiempo para todas las hojas
fig, ax = plt.subplots()
for nombre, r in resultados.items():
    ax.plot(r["t"], np.rad2deg(r["theta"]), label=nombre)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Angulo (grados)")
ax.legend()
ax.grid(True)
plt.show()