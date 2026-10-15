"""
Modelli fisici, propagazione degli errori e fit non lineari
per l'Esercitazione 1 (partitore, RC passa-alto, diodo 1N4148).

Tutte le funzioni sono pure: non dipendono da Streamlit.
Le unità sono SI (V, A, Ω, F, Hz, s) salvo dove indicato.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import odr, stats
from scipy.optimize import curve_fit

# Tensione termica a T = 300 K
VT_300K = 0.02585  # V


@dataclass
class FitResult:
    """Risultato di un fit: parametri, incertezze e statistiche."""

    name: str
    params: dict[str, float] = field(default_factory=dict)
    errors: dict[str, float] = field(default_factory=dict)
    chi2: float = np.nan
    ndof: int = 0
    p_value: float = np.nan

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan


# ---------------------------------------------------------------------------
# TAB 1 - Partitore di tensione
# ---------------------------------------------------------------------------

def ratio_with_error(vo, s_vo, vi, s_vi):
    """Attenuazione sperimentale A = vo/vi con sigma_A = A*sqrt((s_vo/vo)^2 + (s_vi/vi)^2)."""
    vo, s_vo, vi, s_vi = map(np.asarray, (vo, s_vo, vi, s_vi))
    a = vo / vi
    s_a = np.abs(a) * np.sqrt((s_vo / vo) ** 2 + (s_vi / vi) ** 2)
    return a, s_a


def divider_theory(r1, s_r1, r2, s_r2):
    """A_teo = R2/(R1+R2), sigma = sqrt((R2*sR1)^2 + (R1*sR2)^2)/(R1+R2)^2."""
    a = r2 / (r1 + r2)
    s_a = np.sqrt((r2 * s_r1) ** 2 + (r1 * s_r2) ** 2) / (r1 + r2) ** 2
    return a, s_a


def compatibility_t(x1, s1, x2, s2):
    """Fattore t = |x1 - x2| / sqrt(s1^2 + s2^2)."""
    denom = np.sqrt(s1 ** 2 + s2 ** 2)
    return np.abs(x1 - x2) / denom if denom > 0 else np.inf


# ---------------------------------------------------------------------------
# TAB 2 - RC passa-alto
# ---------------------------------------------------------------------------

def rc_cutoff(r, s_r, c, s_c):
    """f_T = 1/(2*pi*R*C), sigma_fT = f_T*sqrt((sR/R)^2 + (sC/C)^2)."""
    f_t = 1.0 / (2.0 * np.pi * r * c)
    s_f_t = f_t * np.sqrt((s_r / r) ** 2 + (s_c / c) ** 2)
    return f_t, s_f_t


def gain_db(h, s_h):
    """G_dB = 20*log10(H), sigma = (20/ln10)*sH/H."""
    h, s_h = np.asarray(h), np.asarray(s_h)
    g = 20.0 * np.log10(h)
    s_g = (20.0 / np.log(10.0)) * s_h / h
    return g, s_g


def phase_deg(f, s_f, dt, s_dt):
    """phi = 360*f*dt (gradi), sigma = 360*sqrt((dt*sf)^2 + (f*sdt)^2)."""
    f, s_f, dt, s_dt = map(np.asarray, (f, s_f, dt, s_dt))
    phi = 360.0 * f * dt
    s_phi = 360.0 * np.sqrt((dt * s_f) ** 2 + (f * s_dt) ** 2)
    return phi, s_phi


def rc_magnitude_model(f, a0, f_t):
    """|H(f)| = A0 / sqrt(1 + (fT/f)^2)."""
    return a0 / np.sqrt(1.0 + (f_t / f) ** 2)


def rc_phase_model_deg(f, f_t):
    """phi(f) = arctan(fT/f), espressa in gradi."""
    return np.degrees(np.arctan(f_t / f))



# ---------------------------------------------------------------------------
# TAB 3 - Diodo 1N4148
# ---------------------------------------------------------------------------

def diode_current_from_voltages(vi, s_vi, vd, s_vd, r, s_r):
    """I_D = (vi - vD)/R con propagazione su vi, vD, R."""
    vi, s_vi, vd, s_vd, r, s_r = map(np.asarray, (vi, s_vi, vd, s_vd, r, s_r))
    vr = vi - vd
    s_vr = np.sqrt(s_vi ** 2 + s_vd ** 2)
    i = vr / r
    s_i = np.sqrt((s_vr / r) ** 2 + (vr * s_r / r ** 2) ** 2)
    return i, s_i


def diode_current_from_resistor(vr, s_vr, r, s_r):
    """I_D = vR/R con propagazione su vR e R."""
    vr, s_vr, r, s_r = map(np.asarray, (vr, s_vr, r, s_r))
    i = vr / r
    s_i = np.sqrt((s_vr / r) ** 2 + (vr * s_r / r ** 2) ** 2)
    return i, s_i


def subtract_series_resistance(vd, s_vd, i_d, s_i_d, r_s, s_r_s):
    """v_D corretta = v_D - R_s*I_D (rimuove la caduta sulla resistenza parassita)."""
    vd, s_vd, i_d, s_i_d = map(np.asarray, (vd, s_vd, i_d, s_i_d))
    v_corr = vd - r_s * i_d
    s_v_corr = np.sqrt(s_vd ** 2 + (i_d * s_r_s) ** 2 + (r_s * s_i_d) ** 2)
    return v_corr, s_v_corr


def shockley_current_ua(v, i_s_ua, eta):
    """I(v) = I_S*(exp(v/(eta*V_T)) - 1), I_S in µA, risultato in µA."""
    arg = np.clip(v / (eta * VT_300K), -80.0, 80.0)
    return i_s_ua * (np.exp(arg) - 1.0)



def _chi2_stats(y, model, sy, n_params):
    chi2 = float(np.sum(((y - model) / sy) ** 2))
    ndof = len(y) - n_params
    p_value = float(stats.chi2.sf(chi2, ndof)) if ndof > 0 else np.nan
    return chi2, ndof, p_value


def fit_rc_magnitude(f, h, s_h, a0_fixed=None):
    """
    Fit del modulo con curve_fit(absolute_sigma=True).
    Se a0_fixed e' un numero, A0 viene fissato; altrimenti e' libero.
    """
    f, h, s_h = map(np.asarray, (f, h, s_h))
    f_t_guess = float(f[np.argmin(np.abs(h - np.max(h) / np.sqrt(2.0)))])

    if a0_fixed is None:
        p0 = [float(np.max(h)), f_t_guess]
        bounds = ([0.0, 1e-3], [np.inf, np.inf])
        popt, pcov = curve_fit(rc_magnitude_model, f, h, p0=p0, sigma=s_h,
                               absolute_sigma=True, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        params = {"A0": popt[0], "fT": popt[1]}
        errors = {"A0": perr[0], "fT": perr[1]}
        n_par = 2
        model = rc_magnitude_model(f, *popt)
    else:
        def func(ff, f_t):
            return rc_magnitude_model(ff, a0_fixed, f_t)

        popt, pcov = curve_fit(func, f, h, p0=[f_t_guess], sigma=s_h,
                               absolute_sigma=True, bounds=([1e-3], [np.inf]))
        perr = np.sqrt(np.diag(pcov))
        params = {"A0": float(a0_fixed), "fT": popt[0]}
        errors = {"A0": 0.0, "fT": perr[0]}
        n_par = 1
        model = func(f, *popt)

    chi2, ndof, p_val = _chi2_stats(h, model, s_h, n_par)
    return FitResult("modulo", params, errors, chi2, ndof, p_val)


def fit_rc_phase(f, phi_deg, s_phi_deg):
    """Fit della fase phi = arctan(fT/f) (dati in gradi) con absolute_sigma=True."""
    f, phi_deg, s_phi_deg = map(np.asarray, (f, phi_deg, s_phi_deg))
    f_t_guess = float(f[np.argmin(np.abs(phi_deg - 45.0))])
    popt, pcov = curve_fit(rc_phase_model_deg, f, phi_deg, p0=[f_t_guess],
                           sigma=s_phi_deg, absolute_sigma=True,
                           bounds=([1e-3], [np.inf]))
    perr = np.sqrt(np.diag(pcov))
    model = rc_phase_model_deg(f, *popt)
    chi2, ndof, p_val = _chi2_stats(phi_deg, model, s_phi_deg, 1)
    return FitResult("fase", {"fT": popt[0]}, {"fT": perr[0]}, chi2, ndof, p_val)


def normalized_residuals(y, model, s_y):
    """Residui normalizzati (y - modello)/sigma."""
    return (np.asarray(y) - np.asarray(model)) / np.asarray(s_y)


def fit_diode_odr(vd, s_vd, i_d, s_i_d, p0=(1e-9, 1.7)):
    """
    Fit di Shockley con scipy.odr (errori su X e Y).
    Internamente la corrente e' espressa in µA per stabilita' numerica.
    p0 = (I_S in A, eta).
    """
    vd, s_vd, i_d, s_i_d = map(np.asarray, (vd, s_vd, i_d, s_i_d))
    y_ua = i_d * 1e6
    s_y_ua = s_i_d * 1e6

    def fcn(beta, x):
        return shockley_current_ua(x, beta[0], beta[1])

    model = odr.Model(fcn)
    data = odr.RealData(vd, y_ua, sx=s_vd, sy=s_y_ua)
    init = [p0[0] * 1e6, p0[1]]
    runner = odr.ODR(data, model, beta0=init, maxit=500)
    out = runner.run()

    is_ua, eta = out.beta
    sd = out.sd_beta
    ndof = len(vd) - 2
    chi2 = float(out.sum_square)
    p_val = float(stats.chi2.sf(chi2, ndof)) if ndof > 0 else np.nan

    return FitResult(
        "ODR Shockley",
        params={"IS": is_ua * 1e-6, "eta": eta},
        errors={"IS": sd[0] * 1e-6, "eta": sd[1]},
        chi2=chi2,
        ndof=ndof,
        p_value=p_val,
    )


def fit_diode_semilog(vd, i_d, s_i_d, v_min=0.0):
    """
    Regressione lineare su ln(I_D) = ln(I_S) + v_D/(eta*V_T)
    usando solo i punti con v_D >= v_min e I_D > 0. Errore su ln(I): sI/I.
    """
    vd, i_d, s_i_d = map(np.asarray, (vd, i_d, s_i_d))
    mask = (vd >= v_min) & (i_d > 0)
    x = vd[mask]
    y = np.log(i_d[mask])
    sy = s_i_d[mask] / i_d[mask]

    def line(xx, m, q):
        return m * xx + q

    popt, pcov = curve_fit(line, x, y, p0=[20.0, np.log(1e-9)],
                           sigma=sy, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    m, q = popt
    s_m, s_q = perr

    i_s = float(np.exp(q))
    s_i_s = i_s * s_q
    eta = 1.0 / (m * VT_300K)
    s_eta = abs(eta) * s_m / abs(m)

    chi2, ndof, p_val = _chi2_stats(y, line(x, *popt), sy, 2)
    return FitResult(
        "semilog",
        params={"IS": i_s, "eta": eta},
        errors={"IS": s_i_s, "eta": s_eta},
        chi2=chi2,
        ndof=ndof,
        p_value=p_val,
    )

