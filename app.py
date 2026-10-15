"""
Interfaccia Streamlit per l'Esercitazione 1 (partitore, RC passa-alto, diodo 1N4148).

Avvio (da terminale, nella cartella del progetto):
    streamlit run app.py

Dipendenze: streamlit, numpy, scipy, matplotlib, pandas, openpyxl, odfpy
(vedi requirements.txt).
Il generatore di tabelle LaTeX e' contenuto in questo file.

Ogni Tab permette di scegliere la sorgente dei dati:
  1. Dati di esempio (simulazione da sample_data.py)
  2. Carica file (.xlsx, .ods, .csv) con template scaricabile
  3. Inserimento / modifica manuale in app
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

import models_and_fits as mf
from models_and_fits import (
    compatibility_t,
    diode_current_from_voltages,
    divider_theory,
    fit_diode_odr,
    fit_diode_semilog,
    fit_rc_magnitude,
    fit_rc_phase,
    gain_db,
    normalized_residuals,
    phase_deg,
    ratio_with_error,
    rc_cutoff,
    shockley_current_ua,
)
from plotting import (
    bode_magnitude_figure,
    bode_phase_figure,
    diode_linear_figure,
    diode_semilog_figure,
    figure_to_pdf_bytes,
)
from sample_data import SEED, diode_example, divider_example, rc_example


SOURCE_SAMPLE = "Dati di esempio (Simulazione)"
SOURCE_UPLOAD = "Carica file (Excel .xlsx, LibreOffice .ods, .csv)"
SOURCE_MANUAL = "Inserimento / Modifica Manuale in App"
SOURCE_OPTIONS = [SOURCE_SAMPLE, SOURCE_UPLOAD, SOURCE_MANUAL]


# ---------------------------------------------------------------------------
# Sorgente dati: selettore, template, caricamento file
# ---------------------------------------------------------------------------

# Colonne dei template per ciascun esperimento (ordine = ordine delle colonne)
DIVIDER_COLUMNS = ["Strumento", "freq_Hz", "Vi_V", "sVi_V", "Vo_V", "sVo_V"]
RC_COLUMNS = ["f_Hz", "sf_Hz", "vi_V", "svi_V", "vo_V", "svo_V", "dt_s", "sdt_s"]
DIODE_COLUMNS = ["vi_V", "svi_V", "vd_V", "svd_V"]


def source_selector(key: str) -> str:
    """Selettore della sorgente dati, visibile in ciascuna Tab."""
    return st.radio("Sorgente dati", SOURCE_OPTIONS, horizontal=True, key=key)


def template_download(columns: list[str], base_name: str, key: str) -> None:
    """Pulsanti per scaricare un template vuoto in formato CSV ed Excel."""
    empty = pd.DataFrame(columns=columns)
    col_csv, col_xlsx = st.columns(2)
    col_csv.download_button(
        "Scarica Template CSV vuoto",
        empty.to_csv(index=False).encode("utf-8"),
        file_name=f"{base_name}_template.csv",
        mime="text/csv",
        key=f"{key}_csv",
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        empty.to_excel(writer, index=False, sheet_name="Dati")
    col_xlsx.download_button(
        "Scarica Template Excel vuoto",
        buf.getvalue(),
        file_name=f"{base_name}_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key}_xlsx",
    )


def load_uploaded_table(uploaded, key: str) -> pd.DataFrame | None:
    """Legge un file caricato (.xlsx, .ods, .csv); per i fogli multipli permette la scelta."""
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        engine = "odf" if name.endswith(".ods") else "openpyxl"
        xls = pd.ExcelFile(uploaded, engine=engine)
        sheets = xls.sheet_names
        sheet = sheets[0]
        if len(sheets) > 1:
            sheet = st.selectbox("Foglio da usare", sheets, key=f"{key}_sheet")
        return pd.read_excel(xls, sheet_name=sheet)
    except Exception as exc:  # file corrotto o formato non valido
        st.error(f"Impossibile leggere il file: {exc}")
        return None


def editable_table(df: pd.DataFrame, key: str, column_config: dict | None = None) -> pd.DataFrame:
    """Tabella modificabile in app (aggiunta/cancellazione righe, correzione celle)."""
    st.caption("Puoi aggiungere, cancellare o correggere le righe direttamente nella tabella.")
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        key=key,
        column_config=column_config,
    )
    return pd.DataFrame(edited)


def number_param(label: str, value: float, key: str, min_value: float | None = None) -> float:
    """Parametro singolo inseribile dall'utente (valore misurato col multimetro)."""
    return float(
        st.number_input(label, value=float(value), min_value=min_value, format="%.6g", key=key)
    )


def clean_numeric(
    df: pd.DataFrame, columns: list[str], required: list[str]
) -> tuple[pd.DataFrame, bool]:
    """Controlla le colonne, converte in numero e scarta le righe senza le misure richieste.

    Le colonne non in `required` (es. freq_Hz per il multimetro) possono essere vuote.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.error(f"Colonne mancanti: {', '.join(missing)}")
        return pd.DataFrame(columns=columns), False
    out = df[columns].copy()
    for c in columns:
        if c not in ("Strumento",):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=required)
    if out.empty:
        st.warning("Nessuna riga valida: compila le misure per proseguire.")
        return out, False
    return out.reset_index(drop=True), True


# ---------------------------------------------------------------------------
# Formattazione e generatore LaTeX
# ---------------------------------------------------------------------------

def fmt_pm(x: float, s: float, tex: bool = False) -> str:
    """Formatta x ± s con 2 cifre significative sull'incertezza.

    Con tex=True restituisce una stringa in modalita' matematica LaTeX ($...$).
    """
    if not (np.isfinite(x) and np.isfinite(s)) or s <= 0:
        body = f"{x:.4g}"
        return f"${body}$" if tex else body

    e = int(np.floor(np.log10(s)))
    pm = " \\pm " if tex else " ± "
    if -4 <= e <= 3:
        dec = max(0, 1 - e)
        body = f"{x:.{dec}f}{pm}{s:.{dec}f}"
    else:
        mant_x = x / 10 ** e
        mant_s = s / 10 ** e
        power = f"\\times 10^{{{e}}}" if tex else f" · 10^{e}"
        body = f"({mant_x:.1f}{pm}{mant_s:.1f}){power}"
    return f"${body}$" if tex else body


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    """Restituisce il codice LaTeX di una tabella (solo \\hline, nessun pacchetto extra)."""
    ncol = len(headers)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "c" * ncol + "}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_section(title, head_plain, head_tex, build_rows, caption, label, key):
    """Mostra una tabella in Streamlit e offre il codice LaTeX esportabile."""
    st.subheader(title)
    st.table([dict(zip(head_plain, r)) for r in build_rows(False)])
    tex = latex_table(head_tex, build_rows(True), caption, label)
    with st.expander("Esporta tabella LaTeX"):
        st.code(tex, language="latex")
        st.download_button(
            "Scarica tabella .tex",
            tex,
            file_name=f"{label.replace(':', '_')}.tex",
            mime="text/plain",
            key=key,
        )


def figure_block(fig, title: str, filename: str, key: str) -> None:
    """Mostra una figura matplotlib e offre il download in PDF vettoriale a 300 dpi."""
    st.pyplot(fig)
    st.download_button(
        f"Scarica {title} (PDF)",
        figure_to_pdf_bytes(fig),
        file_name=filename,
        mime="application/pdf",
        key=key,
    )


# ---------------------------------------------------------------------------
# TAB 1.1 - Partitore di tensione
# ---------------------------------------------------------------------------

DIV_HEAD_PLAIN = ["Strumento", "f [Hz]", "Vi [V]", "Vo [V]", "A sper.", "t vs teorico"]
DIV_HEAD_TEX = [
    "Strumento",
    "$f$ [Hz]",
    "$V_i$ [V]",
    "$V_o$ [V]",
    "$A_{\\mathrm{sper}}$",
    "$t$ vs teorico",
]


def divider_rows(d: dict, tex: bool) -> list[list[str]]:
    a_teo, s_a_teo = divider_theory(d["R1"], d["sR1"], d["R2"], d["sR2"])
    sources = [
        ("Multimetro", None, d["vi_mult"], d["svi_mult"], d["vo_mult"], d["svo_mult"]),
        ("Oscilloscopio", d["freq_osc"], d["vi_osc"], d["svi_osc"], d["vo_osc"], d["svo_osc"]),
    ]
    rows = []
    for name, freqs, vi, svi, vo, svo in sources:
        for k in range(len(vi)):
            a, s_a = ratio_with_error(vo[k], svo[k], vi[k], svi[k])
            t = compatibility_t(float(a), float(s_a), float(a_teo), float(s_a_teo))
            f_txt = "--" if freqs is None else f"{freqs[k]:.0f}"
            rows.append([
                name,
                f_txt,
                fmt_pm(vi[k], svi[k], tex),
                fmt_pm(vo[k], svo[k], tex),
                fmt_pm(a, s_a, tex),
                f"{t:.2f}",
            ])
    return rows


def divider_sample_frame(seed: int) -> tuple[pd.DataFrame, dict]:
    """Converte i dati di esempio nel formato tabellare del template."""
    d = divider_example(seed)
    rows = []
    for name, freqs, vi, svi, vo, svo in [
        ("Multimetro", None, d["vi_mult"], d["svi_mult"], d["vo_mult"], d["svo_mult"]),
        ("Oscilloscopio", d["freq_osc"], d["vi_osc"], d["svi_osc"], d["vo_osc"], d["svo_osc"]),
    ]:
        for k in range(len(vi)):
            rows.append({
                "Strumento": name,
                "freq_Hz": np.nan if freqs is None else float(freqs[k]),
                "Vi_V": float(vi[k]), "sVi_V": float(svi[k]),
                "Vo_V": float(vo[k]), "sVo_V": float(svo[k]),
            })
    params = {"R1": d["R1"], "sR1": d["sR1"], "R2": d["R2"], "sR2": d["sR2"]}
    return pd.DataFrame(rows, columns=DIVIDER_COLUMNS), params


def divider_frame_to_dict(df: pd.DataFrame, params: dict) -> dict:
    """Ricostruisce il dizionario di misure usato dai calcoli a partire dalla tabella."""
    out = dict(params)
    for name, key in [("Multimetro", "mult"), ("Oscilloscopio", "osc")]:
        sub = df[df["Strumento"] == name]
        out[f"vi_{key}"] = sub["Vi_V"].to_numpy(float)
        out[f"svi_{key}"] = sub["sVi_V"].to_numpy(float)
        out[f"vo_{key}"] = sub["Vo_V"].to_numpy(float)
        out[f"svo_{key}"] = sub["sVo_V"].to_numpy(float)
    osc = df[df["Strumento"] == "Oscilloscopio"]
    out["freq_osc"] = osc["freq_Hz"].to_numpy(float)
    return out


def tab_partitore(seed: int) -> None:
    st.header("1.1 Partitore di tensione")
    source = source_selector("src_div")
    sample_df, sample_params = divider_sample_frame(seed)
    params = dict(sample_params)

    st.subheader("Parametri dei resistori")
    c1, c2 = st.columns(2)
    params["R1"] = number_param("R1 [Ω]", params["R1"], "div_R1", min_value=0.0)
    params["sR1"] = number_param("σ R1 [Ω]", params["sR1"], "div_sR1", min_value=0.0)
    params["R2"] = number_param("R2 [Ω]", params["R2"], "div_R2", min_value=0.0)
    params["sR2"] = number_param("σ R2 [Ω]", params["sR2"], "div_sR2", min_value=0.0)
    c1.caption("Valori misurati col multimetro.")
    c2.caption("Incertezze associate.")

    if source == SOURCE_SAMPLE:
        st.caption("Dati di esempio simulati: R1 e R2 con tolleranza 1 %, misure AC.")
        table = editable_table(sample_df, "div_table_sample")
    elif source == SOURCE_UPLOAD:
        template_download(DIVIDER_COLUMNS, "partitore", "div_tpl")
        up = st.file_uploader("Carica il file del partitore", type=["xlsx", "ods", "csv"],
                              key="div_upload")
        loaded = load_uploaded_table(up, "div_upload")
        if loaded is None:
            st.info("Carica un file per visualizzare i dati.")
            return
        table = editable_table(loaded, "div_table_upload")
    else:
        table = editable_table(sample_df.iloc[0:0], "div_table_manual")

    clean, ok = clean_numeric(table, DIVIDER_COLUMNS,
                              required=["Vi_V", "sVi_V", "Vo_V", "sVo_V"])
    if not ok:
        return

    d = divider_frame_to_dict(clean, params)
    a_teo, s_a_teo = divider_theory(d["R1"], d["sR1"], d["R2"], d["sR2"])

    c1, c2, c3 = st.columns(3)
    c1.metric("R1 [Ω]", fmt_pm(d["R1"], d["sR1"]))
    c2.metric("R2 [Ω]", fmt_pm(d["R2"], d["sR2"]))
    c3.metric("A teorico", fmt_pm(a_teo, s_a_teo))

    table_section(
        title="Misure e attenuazione sperimentale",
        head_plain=DIV_HEAD_PLAIN,
        head_tex=DIV_HEAD_TEX,
        build_rows=lambda tex: divider_rows(d, tex),
        caption="Attenuazione del partitore di tensione: misure con multimetro e oscilloscopio",
        label="tab:partitore",
        key="dl_tab_partitore",
    )


# ---------------------------------------------------------------------------
# TAB 1.2 - RC passa-alto
# ---------------------------------------------------------------------------

RC_HEAD_PLAIN = ["f [Hz]", "|H|", "G [dB]", "phi [gradi]"]
RC_HEAD_TEX = ["$f$ [Hz]", "$|H|$", "$G$ [dB]", "$\\varphi$ [$^\\circ$]"]


def rc_compute(d: dict) -> dict:
    f = d["f"]
    h, s_h = ratio_with_error(d["vo"], d["svo"], d["vi"], d["svi"])
    g, s_g = gain_db(h, s_h)
    phi, s_phi = phase_deg(f, d["sf"], d["dt"], d["sdt"])
    return {"f": f, "h": h, "s_h": s_h, "g": g, "s_g": s_g, "phi": phi, "s_phi": s_phi}


def rc_rows(res: dict, tex: bool) -> list[list[str]]:
    rows = []
    for k in range(len(res["f"])):
        rows.append([
            f"{res['f'][k]:.0f}",
            fmt_pm(res["h"][k], res["s_h"][k], tex),
            fmt_pm(res["g"][k], res["s_g"][k], tex),
            fmt_pm(res["phi"][k], res["s_phi"][k], tex),
        ])
    return rows


def rc_sample_frame(seed: int) -> tuple[pd.DataFrame, dict]:
    """Converte i dati di esempio RC nel formato tabellare del template."""
    d = rc_example(seed)
    table = pd.DataFrame({
        "f_Hz": d["f"], "sf_Hz": d["sf"],
        "vi_V": d["vi"], "svi_V": d["svi"],
        "vo_V": d["vo"], "svo_V": d["svo"],
        "dt_s": d["dt"], "sdt_s": d["sdt"],
    }, columns=RC_COLUMNS)
    params = {"R": d["R"], "sR": d["sR"], "C": d["C"], "sC": d["sC"]}
    return table, params


def rc_frame_to_dict(df: pd.DataFrame, params: dict) -> dict:
    """Ricostruisce il dizionario di misure RC dalla tabella modificata."""
    out = dict(params)
    out["f"] = df["f_Hz"].to_numpy(float)
    out["sf"] = df["sf_Hz"].to_numpy(float)
    out["vi"] = df["vi_V"].to_numpy(float)
    out["svi"] = df["svi_V"].to_numpy(float)
    out["vo"] = df["vo_V"].to_numpy(float)
    out["svo"] = df["svo_V"].to_numpy(float)
    out["dt"] = df["dt_s"].to_numpy(float)
    out["sdt"] = df["sdt_s"].to_numpy(float)
    return out


def tab_rc(seed: int) -> None:
    st.header("1.2 RC passa-alto")
    source = source_selector("src_rc")
    sample_df, sample_params = rc_sample_frame(seed)
    params = dict(sample_params)

    st.subheader("Parametri del circuito")
    c1, c2 = st.columns(2)
    params["R"] = number_param("R [Ω]", params["R"], "rc_R", min_value=0.0)
    params["sR"] = number_param("σ R [Ω]", params["sR"], "rc_sR", min_value=0.0)
    params["C"] = number_param("C [F]", params["C"], "rc_C", min_value=0.0)
    params["sC"] = number_param("σ C [F]", params["sC"], "rc_sC", min_value=0.0)
    c1.caption("Valori misurati col multimetro (R) e col capacimetro (C).")
    c2.caption("Incertezze associate.")

    if source == SOURCE_SAMPLE:
        st.caption("Dati di esempio simulati: R ~ 1 kΩ, C ~ 16 nF, rumore strumentale.")
        table = editable_table(sample_df, "rc_table_sample")
    elif source == SOURCE_UPLOAD:
        template_download(RC_COLUMNS, "rc", "rc_tpl")
        up = st.file_uploader("Carica il file del RC passa-alto",
                              type=["xlsx", "ods", "csv"], key="rc_upload")
        loaded = load_uploaded_table(up, "rc_upload")
        if loaded is None:
            st.info("Carica un file per visualizzare i dati.")
            return
        table = editable_table(loaded, "rc_table_upload")
    else:
        table = editable_table(sample_df.iloc[0:0], "rc_table_manual")

    clean, ok = clean_numeric(table, RC_COLUMNS, required=RC_COLUMNS)
    if not ok:
        return

    d = rc_frame_to_dict(clean, params)
    res = rc_compute(d)
    fit_mag = fit_rc_magnitude(res["f"], res["h"], res["s_h"])
    fit_ph = fit_rc_phase(res["f"], res["phi"], res["s_phi"])
    f_t_nom, s_f_t_nom = rc_cutoff(d["R"], d["sR"], d["C"], d["sC"])
    t_mag = compatibility_t(fit_mag.params["fT"], fit_mag.errors["fT"], f_t_nom, s_f_t_nom)
    t_ph = compatibility_t(fit_ph.params["fT"], fit_ph.errors["fT"], f_t_nom, s_f_t_nom)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("fT da R·C [Hz]", fmt_pm(f_t_nom, s_f_t_nom))
    c2.metric("fT fit modulo [Hz]", fmt_pm(fit_mag.params["fT"], fit_mag.errors["fT"]),
              f"t = {t_mag:.2f}")
    c3.metric("fT fit fase [Hz]", fmt_pm(fit_ph.params["fT"], fit_ph.errors["fT"]),
              f"t = {t_ph:.2f}")
    c4.metric("A0 fit modulo", fmt_pm(fit_mag.params["A0"], fit_mag.errors["A0"]))

    def rc_fit_rows(tex: bool) -> list[list[str]]:
        return [
            ["RC nominale", fmt_pm(f_t_nom, s_f_t_nom, tex), "--", "--", "--"],
            [
                "Modulo (A0 libero)",
                fmt_pm(fit_mag.params["fT"], fit_mag.errors["fT"], tex),
                fmt_pm(fit_mag.params["A0"], fit_mag.errors["A0"], tex),
                f"{fit_mag.chi2_red:.2f}",
                f"{fit_mag.p_value:.2f}",
            ],
            [
                "Fase",
                fmt_pm(fit_ph.params["fT"], fit_ph.errors["fT"], tex),
                "--",
                f"{fit_ph.chi2_red:.2f}",
                f"{fit_ph.p_value:.2f}",
            ],
        ]

    table_section(
        title="Misure in frequenza",
        head_plain=RC_HEAD_PLAIN,
        head_tex=RC_HEAD_TEX,
        build_rows=lambda tex: rc_rows(res, tex),
        caption="Modulo e fase del filtro RC passa-alto in funzione della frequenza",
        label="tab:rc_misure",
        key="dl_tab_rc_misure",
    )

    table_section(
        title="Risultati dei fit",
        head_plain=["Metodo", "fT [Hz]", "A0", "chi2 rid.", "p-value"],
        head_tex=["Metodo", "$f_T$ [Hz]", "$A_0$", "$\\chi^2_{\\mathrm{rid}}$", "$p$-value"],
        build_rows=rc_fit_rows,
        caption="Frequenza di taglio dell'RC passa-alto da fit e da valori nominali",
        label="tab:rc_fit",
        key="dl_tab_rc_fit",
    )

    st.subheader("Grafici")
    fig_mag = bode_magnitude_figure(res["f"], res["h"], res["s_h"], fit_mag, db=True)
    figure_block(fig_mag, "Bode modulo", "bode_modulo_rc.pdf", "dl_fig_rc_mag")
    fig_ph = bode_phase_figure(res["f"], res["phi"], res["s_phi"], fit_ph)
    figure_block(fig_ph, "Bode fase", "bode_fase_rc.pdf", "dl_fig_rc_phase")


# ---------------------------------------------------------------------------
# TAB 1.3 - Diodo 1N4148
# ---------------------------------------------------------------------------

DIODE_HEAD_PLAIN = ["Vi [V]", "vD [V]", "I_D [µA]"]
DIODE_HEAD_TEX = ["$V_i$ [V]", "$v_D$ [V]", "$I_D$ [$\\mu$A]"]


def diode_rows(vi, s_vi, vd, s_vd, i_ua, s_i_ua, tex: bool) -> list[list[str]]:
    rows = []
    for k in range(len(vi)):
        rows.append([
            fmt_pm(vi[k], s_vi[k], tex),
            fmt_pm(vd[k], s_vd[k], tex),
            fmt_pm(i_ua[k], s_i_ua[k], tex),
        ])
    return rows


def diode_sample_frame(seed: int) -> tuple[pd.DataFrame, dict]:
    """Converte i dati di esempio del diodo nel formato tabellare del template."""
    d = diode_example(seed)
    table = pd.DataFrame({
        "vi_V": d["vi"], "svi_V": d["svi"],
        "vd_V": d["vd"], "svd_V": d["svd"],
    }, columns=DIODE_COLUMNS)
    params = {"R": d["R"], "sR": d["sR"]}
    return table, params


def tab_diodo(seed: int, v_min: float) -> None:
    st.header("1.3 Diodo 1N4148")
    source = source_selector("src_diode")
    sample_df, sample_params = diode_sample_frame(seed)
    params = dict(sample_params)

    st.subheader("Parametri del circuito")
    c1, c2 = st.columns(2)
    params["R"] = number_param("R serie [Ω]", params["R"], "dio_R", min_value=0.0)
    params["sR"] = number_param("σ R serie [Ω]", params["sR"], "dio_sR", min_value=0.0)
    c1.caption("Resistenza in serie misurata col multimetro.")
    c2.caption("Incertezza associata.")

    if source == SOURCE_SAMPLE:
        st.caption("Dati di esempio simulati: I_S ~ 2.5 nA, η ~ 1.8, R ~ 1 kΩ.")
        table = editable_table(sample_df, "dio_table_sample")
    elif source == SOURCE_UPLOAD:
        template_download(DIODE_COLUMNS, "diodo", "dio_tpl")
        up = st.file_uploader("Carica il file del diodo", type=["xlsx", "ods", "csv"],
                              key="dio_upload")
        loaded = load_uploaded_table(up, "dio_upload")
        if loaded is None:
            st.info("Carica un file per visualizzare i dati.")
            return
        table = editable_table(loaded, "dio_table_upload")
    else:
        table = editable_table(sample_df.iloc[0:0], "dio_table_manual")

    clean, ok = clean_numeric(table, DIODE_COLUMNS, required=DIODE_COLUMNS)
    if not ok:
        return

    d = {"vi": clean["vi_V"].to_numpy(float), "svi": clean["svi_V"].to_numpy(float),
         "vd": clean["vd_V"].to_numpy(float), "svd": clean["svd_V"].to_numpy(float),
         "R": params["R"], "sR": params["sR"]}
    vd, s_vd = d["vd"], d["svd"]
    i_d, s_i_d = diode_current_from_voltages(d["vi"], d["svi"], vd, s_vd, d["R"], d["sR"])
    i_ua, s_i_ua = i_d * 1e6, s_i_d * 1e6

    odr_fit = fit_diode_odr(vd, s_vd, i_d, s_i_d)
    sl_fit = fit_diode_semilog(vd, i_d, s_i_d, v_min=v_min)

    # Curva di Shockley con i parametri ODR (I_S in µA per la funzione di plotting)
    vd_fine = np.linspace(0.0, float(np.max(vd)) * 1.05, 300)
    i_fine_ua = shockley_current_ua(vd_fine, odr_fit.params["IS"] * 1e6, odr_fit.params["eta"])

    # Residui del fit semilogaritmico sui soli punti usati nel fit
    mask = (vd >= v_min) & (i_d > 0)
    vd_fit = vd[mask]
    i_fit = i_d[mask]
    s_fit = s_i_d[mask]
    ln_model = np.log(sl_fit.params["IS"]) + vd_fit / (sl_fit.params["eta"] * mf.VT_300K)
    res_fit = normalized_residuals(np.log(i_fit), ln_model, s_fit / i_fit)
    # Retta semilog in ln(µA), coerente con l'asse del grafico
    ln_i_fine = np.log(sl_fit.params["IS"] * 1e6) + vd_fine / (sl_fit.params["eta"] * mf.VT_300K)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("I_S ODR [nA]", fmt_pm(odr_fit.params["IS"] * 1e9, odr_fit.errors["IS"] * 1e9))
    c2.metric("η ODR", fmt_pm(odr_fit.params["eta"], odr_fit.errors["eta"]))
    c3.metric("I_S semilog [nA]", fmt_pm(sl_fit.params["IS"] * 1e9, sl_fit.errors["IS"] * 1e9))
    c4.metric("η semilog", fmt_pm(sl_fit.params["eta"], sl_fit.errors["eta"]))

    def diode_fit_rows(tex: bool) -> list[list[str]]:
        rows = []
        for label, fit in [("ODR (X e Y)", odr_fit), ("Semilogaritmico", sl_fit)]:
            rows.append([
                label,
                fmt_pm(fit.params["IS"] * 1e9, fit.errors["IS"] * 1e9, tex),
                fmt_pm(fit.params["eta"], fit.errors["eta"], tex),
                f"{fit.chi2_red:.2f}",
                f"{fit.p_value:.2f}",
            ])
        return rows

    table_section(
        title="Misure della caratteristica I-V",
        head_plain=DIODE_HEAD_PLAIN,
        head_tex=DIODE_HEAD_TEX,
        build_rows=lambda tex: diode_rows(d["vi"], d["svi"], vd, s_vd, i_ua, s_i_ua, tex),
        caption="Caratteristica I-V del diodo 1N4148 con corrente ricavata dalla resistenza in serie",
        label="tab:diodo_misure",
        key="dl_tab_diodo_misure",
    )

    table_section(
        title="Risultati dei fit",
        head_plain=["Metodo", "I_S [nA]", "eta", "chi2 rid.", "p-value"],
        head_tex=["Metodo", "$I_S$ [nA]", "$\\eta$", "$\\chi^2_{\\mathrm{rid}}$", "$p$-value"],
        build_rows=diode_fit_rows,
        caption="Parametri del modello di Shockley da fit ODR e da regressione semilogaritmica",
        label="tab:diodo_fit",
        key="dl_tab_diodo_fit",
    )

    st.subheader("Grafici")
    fig_lin = diode_linear_figure(vd, s_vd, i_ua, s_i_ua, vd_fine, i_fine_ua)
    figure_block(fig_lin, "grafico I-V lineare", "diodo_iv_lineare.pdf", "dl_fig_diodo_lin")
    fig_semi = diode_semilog_figure(vd, i_ua, s_i_ua, vd_fit, res_fit, vd_fine, ln_i_fine)
    figure_block(fig_semi, "grafico I-V semilogaritmico", "diodo_iv_semilog.pdf", "dl_fig_diodo_semi")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Esercitazione 1 - Lab SS", layout="wide")
    st.title("Esercitazione 1 - Analisi dati")

    st.sidebar.header("Impostazioni dati di esempio")
    st.sidebar.caption("Seed e V_min valgono solo per la sorgente 'Dati di esempio'.")
    seed = int(st.sidebar.number_input("Seed generatore", value=SEED, step=1))
    v_min = st.sidebar.slider("V_min per fit semilogaritmico [V]", 0.0, 0.7, 0.4, 0.05)

    tab1, tab2, tab3 = st.tabs(["1.1 Partitore", "1.2 RC passa-alto", "1.3 Diodo"])
    with tab1:
        tab_partitore(seed)
    with tab2:
        tab_rc(seed)
    with tab3:
        tab_diodo(seed, v_min)


main()
