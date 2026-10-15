"""
Grafici scientifici per l'Esercitazione 1.

Le figure sono costruite con matplotlib.figure.Figure (senza pyplot):
non c'e' stato globale e il codice e' sicuro dentro Streamlit.
Ogni figura ha due pannelli impilati: grafico principale e residui normalizzati.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

import models_and_fits as mf

COLOR_DATA = "#1f4e79"
COLOR_FIT = "#c00000"
COLOR_ASYM = "#7f7f7f"
DPI_EXPORT = 300


def _style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)


def _residual_panel(ax, f, y, s_y, model) -> None:
    """Pannello inferiore: residui normalizzati (y - modello)/sigma."""
    res = mf.normalized_residuals(y, model, s_y)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-1, 1, color="green", alpha=0.08)
    ax.plot(f, res, "o", color=COLOR_DATA, ms=4)
    ax.set_ylabel("Residui norm.")
    ax.set_xlabel("Frequenza f [Hz]")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)


def bode_magnitude_figure(f, h, s_h, fit: mf.FitResult, db: bool = True) -> Figure:
    """
    Bode del modulo: dati con barre d'errore, best-fit, asintoti
    (+20 dB/dec per f << fT e 0 dB per f >> fT), verticale a fT con
    indicazione del punto a -3 dB. Pannello inferiore: residui normalizzati.
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=float)
    s_h = np.asarray(s_h, dtype=float)
    a0 = fit.params["A0"]
    f_t = fit.params["fT"]

    f_fine = np.logspace(np.log10(f.min()) - 0.3, np.log10(f.max()) + 0.3, 400)
    h_fit_fine = mf.rc_magnitude_model(f_fine, a0, f_t)
    h_fit = mf.rc_magnitude_model(f, a0, f_t)

    fig = Figure(figsize=(7, 7), dpi=DPI_EXPORT)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    if db:
        y, s_y = mf.gain_db(h, s_h)
        y_fit_fine = 20 * np.log10(h_fit_fine)
        y_fit = 20 * np.log10(h_fit)
        asym_low = 20 * np.log10(a0) + 20 * np.log10(f_fine / f_t)
        asym_high = np.full_like(f_fine, 20 * np.log10(a0))
        y_fT = 20 * np.log10(a0) - 3.01
        ylabel = r"$G$ [dB]"
    else:
        y, s_y = h, s_h
        y_fit_fine, y_fit = h_fit_fine, h_fit
        asym_low = a0 * f_fine / f_t
        asym_high = np.full_like(f_fine, a0)
        y_fT = a0 / np.sqrt(2.0)
        ylabel = r"$|H(f)|$"

    ax.errorbar(f, y, yerr=s_y, fmt="o", color=COLOR_DATA, ms=4, capsize=2,
                label="Dati")
    ax.plot(f_fine, y_fit_fine, color=COLOR_FIT, lw=1.6, label="Best-fit")
    ax.plot(f_fine, asym_low, "--", color=COLOR_ASYM, lw=1,
            label="Asintoti")
    ax.plot(f_fine, asym_high, "--", color=COLOR_ASYM, lw=1)
    ax.axvline(f_t, color="k", ls=":", lw=1, label=f"fT = {f_t:.0f} Hz")
    ax.plot([f_t], [y_fT], "s", color="k", ms=4,
            label="-3 dB" if db else "1/sqrt(2)")
    _style(ax, ylabel)
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Diagramma di Bode - modulo")
    ax.tick_params(labelbottom=False)

    _residual_panel(ax_r, f, y, s_y, y_fit)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.09)
    return fig


def bode_phase_figure(f, phi_deg, s_phi_deg, fit: mf.FitResult) -> Figure:
    """
    Bode della fase: dati con barre d'errore, curva di fit, asintoti 90 gradi (f << fT)
    e 0 gradi (f >> fT), passaggio a 45 gradi in fT. Pannello inferiore: residui normalizzati.
    """
    f = np.asarray(f, dtype=float)
    phi = np.asarray(phi_deg, dtype=float)
    s_phi = np.asarray(s_phi_deg, dtype=float)
    f_t = fit.params["fT"]

    f_fine = np.logspace(np.log10(f.min()) - 0.3, np.log10(f.max()) + 0.3, 400)
    phi_fit_fine = mf.rc_phase_model_deg(f_fine, f_t)
    phi_fit = mf.rc_phase_model_deg(f, f_t)

    fig = Figure(figsize=(7, 7), dpi=DPI_EXPORT)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(f, phi, yerr=s_phi, fmt="o", color=COLOR_DATA, ms=4, capsize=2,
                label="Dati")
    ax.plot(f_fine, phi_fit_fine, color=COLOR_FIT, lw=1.6, label="Best-fit")
    ax.axhline(90.0, ls="--", color=COLOR_ASYM, lw=1, label="Asintoti 90/0 gradi")
    ax.axhline(0.0, ls="--", color=COLOR_ASYM, lw=1)
    ax.axhline(45.0, ls=":", color="k", lw=0.8)
    ax.axvline(f_t, color="k", ls=":", lw=1, label=f"fT = {f_t:.0f} Hz")
    _style(ax, r"Fase $\phi$ [gradi]")
    ax.set_xscale("log")
    ax.set_ylim(-5, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Diagramma di Bode - fase")
    ax.tick_params(labelbottom=False)

    _residual_panel(ax_r, f, phi, s_phi, phi_fit)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.09)
    return fig


def diode_linear_figure(vd, s_vd, i_ua, s_i_ua, vd_fine, i_fine_ua) -> Figure:
    """
    Grafico lineare I_D (uA) vs v_D (V) con curva di fit e residui normalizzati.
    """
    vd = np.asarray(vd, dtype=float)
    i_ua = np.asarray(i_ua, dtype=float)
    s_vd = np.asarray(s_vd, dtype=float)
    s_i_ua = np.asarray(s_i_ua, dtype=float)

    fig = Figure(figsize=(7, 7), dpi=DPI_EXPORT)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(vd, i_ua, xerr=s_vd, yerr=s_i_ua, fmt="o", color=COLOR_DATA,
                ms=4, capsize=2, label="Dati")
    ax.plot(vd_fine, i_fine_ua, color=COLOR_FIT, lw=1.6, label="Fit Shockley")
    _style(ax, r"$I_D$ [$\mu$A]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Caratteristica I-V del diodo (lineare)")
    ax.tick_params(labelbottom=False)

    ax_r.axhline(0.0, color="k", lw=0.8)
    ax_r.set_xlabel(r"$v_D$ [V]")
    ax_r.set_ylabel("Residui norm.")
    ax_r.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.09)
    return fig


def diode_semilog_figure(vd, i_ua, s_i_ua, vd_fit, res_fit, vd_fine,
                         lnI_fine) -> Figure:
    """
    Grafico semilogaritmico ln(I_D) vs v_D con fit lineare nel tratto esponenziale.
    res_fit: residui normalizzati del fit lineare, calcolati sui punti vd_fit.
    """
    vd = np.asarray(vd, dtype=float)
    i_ua = np.asarray(i_ua, dtype=float)
    s_i_ua = np.asarray(s_i_ua, dtype=float)
    ln_i = np.log(i_ua)
    s_ln_i = s_i_ua / i_ua

    fig = Figure(figsize=(7, 7), dpi=DPI_EXPORT)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(vd, ln_i, yerr=s_ln_i, fmt="o", color=COLOR_DATA, ms=4,
                capsize=2, label="Dati")
    ax.plot(vd_fine, lnI_fine, color=COLOR_FIT, lw=1.6,
            label="Fit lineare (tratto esponenziale)")
    _style(ax, r"$\ln(I_D)$ [$\ln\mu$A]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Caratteristica I-V del diodo (semilogaritmica)")
    ax.tick_params(labelbottom=False)

    ax_r.axhline(0.0, color="k", lw=0.8)
    ax_r.plot(vd_fit, res_fit, "o", color=COLOR_DATA, ms=4)
    ax_r.set_xlabel(r"$v_D$ [V]")
    ax_r.set_ylabel("Residui norm.")
    ax_r.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.09)
    return fig


def figure_to_pdf_bytes(fig: Figure) -> bytes:
    """Esporta una figura in PDF vettoriale a 300 dpi (bytes)."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=DPI_EXPORT, bbox_inches="tight")
    return buf.getvalue()

