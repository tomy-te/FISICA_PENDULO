from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------- CONFIGURACION ----------------
ARCHIVO = "Pendulo 5.xlsx"   # ruta al Excel
LONGITUD = 0.405             # m. Es el radio que usaron en el Excel (Vt = omega * r)
G = 9.81                     # m/s^2
VERTICAL_DEG = -90.0         # theta del Excel vale -90 cuando el pendulo esta vertical
VENTANA = 9                  # impar. Ventana (en cuadros) del suavizado de derivadas
CARPETA_SALIDA = Path("resultados")
# -----------------------------------------------


def leer_bloques(archivo):
    """Devuelve {amplitud_nominal: DataFrame} con una tabla limpia por cada bloque."""
    raw = pd.read_excel(archivo, header=None)
    titulos = raw.iloc[0]
    bloques = {}
    for c, txt in titulos.items():
        if isinstance(txt, str) and "grados" in txt.lower():
            amp = float(txt.lower().replace("grados", "").strip())
            cols = {
                "t": c - 5, "v": c - 4, "theta_deg": c - 3, "omega_deg": c - 2, "alfa_deg": c - 1,
                "theta": c + 1, "omega": c + 2, "alfa": c + 3, "Vt": c + 4, "At": c + 5, "An": c + 6,
            }
            d = pd.DataFrame({k: pd.to_numeric(raw.iloc[2:, i], errors="coerce") for k, i in cols.items()})
            d = d.dropna(subset=["t", "theta_deg"]).reset_index(drop=True)
            bloques[amp] = d
    return bloques


def procesar(d):
    """Agrega el angulo respecto de la vertical y derivadas suavizadas."""
    d = d.copy()
    t = d["t"].to_numpy()
    dt = np.median(np.diff(t))
    # Angulo medido desde la vertical (0 = pendulo colgando recto)
    d["phi_deg"] = d["theta_deg"] - VERTICAL_DEG
    phi = np.deg2rad(d["phi_deg"].to_numpy())

    # El tracking es ruidoso: se suaviza con Savitzky-Golay antes de derivar
    v = min(VENTANA, len(d) - (1 - len(d) % 2))
    d["phi_s"] = savgol_filter(phi, v, 3)
    d["omega_s"] = savgol_filter(phi, v, 3, deriv=1, delta=dt)   # rad/s
    d["alfa_s"] = savgol_filter(phi, v, 3, deriv=2, delta=dt)    # rad/s^2
    # Magnitudes lineales / de movimiento circular
    d["Vt_s"] = LONGITUD * d["omega_s"]                           # velocidad tangencial
    d["At_s"] = LONGITUD * d["alfa_s"]                            # aceleracion tangencial
    d["An_s"] = LONGITUD * d["omega_s"] ** 2                      # aceleracion normal (centripeta)
    return d


def periodo(d):
    """Periodo T a partir de los cruces por cero (misma direccion) con interpolacion lineal."""
    t = d["t"].to_numpy()
    y = d["phi_s"].to_numpy() - np.median(d["phi_s"])
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]            # cruces ascendentes
    tc = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    if len(tc) < 2:
        return np.nan, np.nan, len(tc)
    T = np.diff(tc)
    return T.mean(), T.std(ddof=1) if len(T) > 1 else np.nan, len(tc)


def main():
    CARPETA_SALIDA.mkdir(exist_ok=True)
    bloques = leer_bloques(ARCHIVO)
    datos = {amp: procesar(d) for amp, d in bloques.items()}

    T0 = 2 * np.pi * np.sqrt(LONGITUD / G)
    filas = []
    for amp, d in datos.items():
        T, sT, n = periodo(d)
        a_med = (d["phi_s"].max() - d["phi_s"].min()) / 2          # amplitud medida (rad)
        T_corr = T0 * (1 + a_med ** 2 / 16)                         # 1ra correccion por amplitud
        filas.append({
            "amplitud nominal (grados)": amp,
            "amplitud medida (grados)": np.rad2deg(a_med),
            "cuadros": len(d),
            "duracion (s)": d["t"].max() - d["t"].min(),
            "T medido (s)": T,
            "desvio T (s)": sT,
            "cruces": n,
            "T0 pequeñas oscil. (s)": T0,
            "T corregido (s)": T_corr,
            "T medido / T0": T / T0,
        })
        # Un CSV por amplitud con todo lo calculado
        d.to_csv(CARPETA_SALIDA / f"pendulo_{int(amp)}grados.csv", index=False)

    resumen = pd.DataFrame(filas)
    resumen.to_csv(CARPETA_SALIDA / "resumen.csv", index=False)
    print(f"L = {LONGITUD} m  ->  T0 = 2*pi*sqrt(L/g) = {T0:.3f} s\n")
    print(resumen.round(3).to_string(index=False))

    # ---- Grafico 1: posicion angular vs tiempo, todas las amplitudes ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for amp, d in datos.items():
        ax.plot(d["t"], d["phi_deg"], ".", ms=3, alpha=0.4)
        ax.plot(d["t"], np.rad2deg(d["phi_s"]), lw=1.5, label=f"{amp:g}°")
    ax.set(xlabel="Tiempo (s)", ylabel="Ángulo respecto de la vertical (°)",
           title="Posición angular vs tiempo")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(title="Amplitud")
    fig.tight_layout()
    fig.savefig(CARPETA_SALIDA / "theta_vs_t.png", dpi=150)

    # ---- Grafico 2: una figura por amplitud con theta, omega, alfa, Vt, At, An ----
    for amp, d in datos.items():
        fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        paneles = [
            ("phi_deg", "θ (°)", None), ("omega_s", "ω (rad/s)", "omega"),
            ("alfa_s", "α (rad/s²)", "alfa"), ("Vt_s", "Vt (m/s)", "Vt"),
            ("At_s", "At (m/s²)", "At"), ("An_s", "An (m/s²)", "An"),
        ]
        for ax, (col, lab, col_excel) in zip(axs.T.ravel(), paneles):
            if col == "phi_deg":
                ax.plot(d["t"], d["phi_deg"], ".", ms=3, alpha=0.5, label="medido")
                ax.plot(d["t"], np.rad2deg(d["phi_s"]), lw=1.5, label="suavizado")
            else:
                if col_excel:
                    ax.plot(d["t"], d[col_excel], ".", ms=3, alpha=0.3, label="Excel (sin suavizar)")
                ax.plot(d["t"], d[col], lw=1.5, label="suavizado")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axs[-1, 0].set_xlabel("Tiempo (s)")
        axs[-1, 1].set_xlabel("Tiempo (s)")
        fig.suptitle(f"Péndulo - amplitud {amp:g}°")
        fig.tight_layout()
        fig.savefig(CARPETA_SALIDA / f"analisis_{int(amp)}grados.png", dpi=150)

    # ---- Grafico 3: periodo vs amplitud (chequeo fisico) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(resumen["amplitud medida (grados)"], resumen["T medido (s)"],
                yerr=resumen["desvio T (s)"].fillna(0), fmt="o", capsize=3, label="T medido")
    a = np.linspace(0, resumen["amplitud medida (grados)"].max() * 1.1, 100)
    ax.plot(a, T0 * (1 + np.deg2rad(a) ** 2 / 16), "--", label="T0·(1 + θ0²/16)")
    ax.axhline(T0, color="gray", lw=0.8, label="T0 (ángulo chico)")
    ax.set(xlabel="Amplitud (°)", ylabel="Período (s)", title="Período vs amplitud")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA_SALIDA / "periodo_vs_amplitud.png", dpi=150)

    print(f"\nResultados guardados en: {CARPETA_SALIDA.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()