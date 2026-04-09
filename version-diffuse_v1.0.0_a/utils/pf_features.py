"""
Shared utilities for atmospheric neutrino passing fraction calculations.

This module is intentionally:
- stateless
- side-effect free
- importable from anywhere

It must NOT contain:
- IceTray modules
- frame logic
- file I/O
- configuration paths
"""

import numpy as np
from scipy.special import erfc
from icecube import dataclasses, MuonGun



def getDepth(p):
    """
    Compute detector entry depth for a particle.

    Parameters
    ----------
    p : I3Particle

    Returns
    -------
    float
        Depth in meters.
        Returns 0.0 if no intersection is found.
    """
    surface_det = MuonGun.Cylinder(
        1200, 700, dataclasses.I3Position(0, 0, -100)
    )

    intersection = surface_det.intersection(p.pos, p.dir)
    if np.isfinite(intersection.first):
        z_inter = p.pos.z - intersection.first * np.cos(p.dir.zenith)
        return 1948.07 - z_inter

    return 0.0


def getDetectorTime(p):
    """
    Compute the detector entry time for a particle.

    Parameters
    ----------
    p : I3Particle

    Returns
    -------
    float
        Time in ns. Returns np.inf if no intersection is found.
    """
    surface_det = MuonGun.Cylinder(
        1200, 700, dataclasses.I3Position(0, 0, -100)
    )

    intersection = surface_det.intersection(p.pos, p.dir)
    if np.isfinite(intersection.first):
        return p.time + intersection.first / dataclasses.I3Constants.c

    return np.inf


def p_pass(raw_score):
    """
    Logistic transform of raw BDT score.

    Parameters
    ----------
    raw_score : float or array-like

    Returns
    -------
    float or ndarray
        Pass probability.
    """
    return 1.0 / (1.0 + np.exp(-raw_score))


def p_light(p_pass_event):
    """
    Convert pass probability to light probability.

    Parameters
    ----------
    p_pass_event : float or array-like

    Returns
    -------
    float or ndarray
        Light probability.
    """
    return 1.0 - p_pass_event


def p_light_per_event(event_row, bdt, feature_names):
    """
    Build a callable p_light(E_mu) function for a single event.

    Parameters
    ----------
    event_row : pandas.Series
        Event features.
    bdt : lightgbm.Booster
        Trained BDT model.
    feature_names : list[str]
        Feature order expected by the BDT.

    Returns
    -------
    callable
        Function mapping muon bundle energy -> p_light
    """
    base_vals = event_row[feature_names].values.astype(float)
    muon_idx = feature_names.index("muonbundle_energy")

    def get_p_light(E_mu):
        E_mu_arr = np.atleast_1d(E_mu).astype(float)
        X = np.tile(base_vals, (len(E_mu_arr), 1))
        X[:, muon_idx] = E_mu_arr
        raw = bdt.predict(X)
        return p_light(p_pass(raw))

    return get_p_light


def erfc_fit(x, a, b, c):
    """
    Complementary error function model used for PF fits.

    Parameters
    ----------
    x : array-like
        Shift values.
    a, b, c : float
        Fit parameters.

    Returns
    -------
    ndarray
        Model values.
    """
    z = (x - b) / (c * np.sqrt(2))
    return a * (2.0 - erfc(z))
