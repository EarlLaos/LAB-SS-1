"""
Calcolatore automatico dell'incertezza strumentale.

Modello:  sigma = (percentuale/100) * lettura + risoluzione
Esempio:  sigma_V = 1 % * V + 1 digit
"""

from __future__ import annotations

import numpy as np


def instrument_sigma(reading, pct: float, resolution: float):
    """
    sigma = pct/100 * |lettura| + risoluzione.
    Accetta scalari o array NumPy; restituisce lo stesso tipo di input.
    """
    reading_arr = np.asarray(reading, dtype=float)
    sigma = (pct / 100.0) * np.abs(reading_arr) + resolution
    return sigma if sigma.ndim else float(sigma)
