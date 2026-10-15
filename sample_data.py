"""
Generatori di dati di esempio realistici per il test dell'app.

Il rumore gaussiano e' coerente con l'incertezza strumentale:
ogni misura viene perturbata con sigma = pct*lettura + risoluzione.
Valori di default fissati da seed per risultati riproducibili.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from instruments import instrument_sigma

SEED = 42


def divider_example(seed: int = SEED) -> dict:
    """Partitore: R1, R2 con tolleranza 1 % e tensioni multimetro/oscilloscopio."""
    rng = np.random.default_rng(seed)
    r1_nom, r2_nom = 10_000.0, 4_700.0
    s_r1, s_r2 = 0.01 * r1_nom, 0.01 * r2_nom
    r1 = r1_nom + rng.normal(0, s_r1)
    r2 = r2_nom + rng.normal(0, s_r2)

    a_true = r2 / (r1 + r2)
    vi_true = np.array([1.00, 2.00, 5.00])  # V efficaci, multimetro AC
    vi = vi_true + rng.normal(0, instrument_sigma(vi_true, 0.5, 0.01))
    vo_true = a_true * vi_true
    vo = vo_true + rng.normal(0, instrument_sigma(vo_true, 0.5, 0.005))
    s_vi_mult = instrument_sigma(vi, 0.5, 0.01)
    s_vo_mult = instrument_sigma(vo, 0.5, 0.005)

    # Oscilloscopio: tre frequenze, tensione fissa a 2 V, rumore maggiore
    freqs = np.array([100.0, 1_000.0, 10_000.0])
    vi_osc = 2.00 + rng.normal(0, 0.02, 3)
    vo_osc = a_true * 2.00 + rng.normal(0, 0.02, 3)

    return {
        "R1": r1, "sR1": s_r1, "R2": r2, "sR2": s_r2,
        "vi_mult": vi, "svi_mult": s_vi_mult,
        "vo_mult": vo, "svo_mult": s_vo_mult,
        "freq_osc": freqs,
        "vi_osc": vi_osc, "svi_osc": np.full(3, 0.02),
        "vo_osc": vo_osc, "svo_osc": np.full(3, 0.02),
    }


def rc_example(seed: int = SEED) -> dict:
    """RC passa-alto: R ~ 1 kOhm, C ~ 16 nF (fT ~ 10 kHz), rumore strumentale."""
    rng = np.random.default_rng(seed)
    r_nom, c_nom = 1_000.0, 16e-9
    r = r_nom * (1 + rng.normal(0, 0.01))
    c = c_nom * (1 + rng.normal(0, 0.02))
    s_r, s_c = 0.01 * r_nom, 0.02 * c_nom

    f = np.logspace(2, 5, 18)  # 100 Hz - 100 kHz
    ft = 1.0 / (2 * np.pi * r * c)
    h_true = 1.0 / np.sqrt(1 + (ft / f) ** 2)  # modulo passa-alto, A0 = 1
    vi = np.full_like(f, 1.0)
    s_vi = instrument_sigma(vi, 0.5, 0.005)
    vo = h_true + rng.normal(0, instrument_sigma(h_true, 0.5, 0.005))
    s_vo = instrument_sigma(vo, 0.5, 0.005)

    phi_true = np.degrees(np.arctan(ft / f))
    dt_true = phi_true / (360.0 * f)
    s_dt = np.full_like(f, 2e-7)
    dt = dt_true + rng.normal(0, s_dt)
    s_f = 0.001 * f

    return {
        "R": r, "sR": s_r, "C": c, "sC": s_c,
        "f": f, "sf": s_f,
        "vi": vi, "svi": s_vi,
        "vo": vo, "svo": s_vo,
        "dt": dt, "sdt": s_dt,
    }


def diode_example(seed: int = SEED) -> dict:
    """Diodo 1N4148 simulato: I_S ~ 2.5 nA, eta ~ 1.8, R ~ 1 kOhm."""
    rng = np.random.default_rng(seed)
    r = 1_000.0 * (1 + rng.normal(0, 0.01))
    s_r = 0.01 * 1_000.0
    i_s_true, eta_true = 2.5e-9, 1.8
    vt = 0.02585

    vi = np.linspace(0.3, 3.0, 25)
    s_vi = instrument_sigma(vi, 0.5, 0.005)

    # Per ogni vi si risolve numericamente vD tale che (vi - vD)/R = I_S(exp(vD/(eta VT)) - 1)
    vd_true = np.empty_like(vi)
    for k, v in enumerate(vi):
        def g(vd, v=v):
            i = i_s_true * (np.exp(vd / (eta_true * vt)) - 1.0)
            return (v - vd) / r - i
        vd_true[k] = brentq(g, 0.0, v - 1e-9)

    vd = vd_true + rng.normal(0, instrument_sigma(vd_true, 0.2, 0.001))
    s_vd = instrument_sigma(vd, 0.2, 0.001)
    vr = vi - vd
    s_vr = np.sqrt(s_vi ** 2 + s_vd ** 2)
    i_d = vr / r
    s_i_d = np.sqrt((s_vr / r) ** 2 + (vr * s_r / r ** 2) ** 2)

    return {
        "R": r, "sR": s_r,
        "vi": vi, "svi": s_vi,
        "vd": vd, "svd": s_vd,
        "id": i_d, "sid": s_i_d,
    }
