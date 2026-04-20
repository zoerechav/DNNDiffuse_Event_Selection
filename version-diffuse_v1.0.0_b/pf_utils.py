import numpy as np
import pickle
from scipy.special import erfc
from icecube import dataclasses, MuonGun
from scipy.interpolate import interp1d


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

    event_row must already be ordered according to feature_names.
    """

  
    base_vals = np.asarray(event_row, dtype=float)

 
    if base_vals.ndim != 1:
        base_vals = base_vals.flatten()

    muon_idx = feature_names.index("muonbundle_energy")

    def get_p_light(E_mu):
        E_mu_arr = np.atleast_1d(E_mu).astype(float)

        X = np.empty((len(E_mu_arr), base_vals.size), dtype=float)
        X[:] = base_vals

        X[:, muon_idx] = E_mu_arr

        raw = bdt.predict(X)
        return p_light(p_pass(raw))

    return get_p_light

def smooth_blend(E_GeV, p_low, p_high, E0_GeV=1e6, transition_width_dex=0.2):
    """
    Controlled sigmoid blend centered at E0 (default 1 PeV),
    with pure-low below band and pure-high above band.
    
    To blend together the high energy and low energy bdt around their transition point at 1PeV
    purely for smoothing purposes, no physics implementation
    
    0.2 transition width was chosen after a grid scan optimization study
    """
    E_GeV = np.asarray(E_GeV, dtype=float)
    p_low = np.asarray(p_low, dtype=float)
    p_high = np.asarray(p_high, dtype=float)
    
    E_GeV = np.clip(E_GeV, 1e-10, None)
    
    p_low  = np.clip(p_low,  0.0, 1.0)
    p_high = np.clip(p_high, 0.0, 1.0)

    logE  = np.log10(E_GeV)
    logE0 = np.log10(E0_GeV)

    lower = logE0 - transition_width_dex
    upper = logE0 + transition_width_dex

    w = np.zeros_like(logE)

    in_band = (logE >= lower) & (logE <= upper)
    x = (logE[in_band] - logE0) / transition_width_dex
    w[in_band] = 1.0 / (1.0 + np.exp(-x))  # sigmoid

    w[logE > upper] = 1.0  # pure high
    w[logE < lower] = 0.0  # pure low

    p_blend = (1.0 - w) * p_low + w * p_high
    return p_blend, w


def build_sigma_of_p_from_cv(
    cv_pkl_path,
    n_p_bins=50,
    min_count=100,
):
    """
    Build σ(p_light): the typical (median) CV standard deviation
    as a function of p_light.
    """
    with open(cv_pkl_path, "rb") as f:
        cv = pickle.load(f)

    p_cv = np.asarray(cv["p_light_cv_median"])
    s_cv = np.asarray(cv["p_light_cv_std"])

    mfin = np.isfinite(p_cv) & np.isfinite(s_cv)
    p_cv = p_cv[mfin]
    s_cv = s_cv[mfin]

    bins = np.linspace(0, 1, n_p_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    sig_med = np.full_like(centers, np.nan, dtype=float)

    for i in range(len(bins) - 1):
        m = (p_cv >= bins[i]) & (p_cv < bins[i+1])
        if m.sum() >= min_count:
            sig_med[i] = np.median(s_cv[m])   # ← THIS is σ(p)

    good = np.isfinite(sig_med)
    if good.sum() < 5:
        raise RuntimeError("Not enough populated p-bins.")

    return interp1d(
        centers[good],
        sig_med[good],
        bounds_error=False,
        fill_value=(sig_med[good][0], sig_med[good][-1]),
    )

def p_light_per_event_with_uncertainty_blended(
    event_row,
    bdt_low,
    bdt_high,
    feature_names,
    sigma_of_p_low,
    sigma_of_p_high,
    E0_GeV=1e6,
    transition_width_dex=0.2,
):
    """
    Blended p_light and uncertainty using quadrature
    combination of uncertainties.
    """

    # --- nominal curves ---
    p_nom_low  = p_light_per_event(event_row, bdt_low,  feature_names)
    p_nom_high = p_light_per_event(event_row, bdt_high, feature_names)

    def p_nom(E_mu):
        p_low  = np.asarray(p_nom_low(E_mu))
        p_high = np.asarray(p_nom_high(E_mu))

        p_blend, w = smooth_blend(
            E_mu,
            p_low,
            p_high,
            E0_GeV=E0_GeV,
            transition_width_dex=transition_width_dex,
        )
        
        p_blend = np.nan_to_num(p_blend, nan=0.0, posinf=1.0, neginf=0.0)
        return p_blend

    def sigma(E_mu):
        E_mu = np.asarray(E_mu)

        p_low  = np.asarray(p_nom_low(E_mu))
        p_high = np.asarray(p_nom_high(E_mu))

        s_low  = np.asarray(sigma_of_p_low(np.clip(p_low, 0, 1)))
        s_high = np.asarray(sigma_of_p_high(np.clip(p_high, 0, 1)))

        # Get blend weight only
        _, w = smooth_blend(
            E_mu,
            p_low,
            p_high,
            E0_GeV=E0_GeV,
            transition_width_dex=transition_width_dex,
        )

        # Quadrature combination of weighted variances
        var_blend = (1.0 - w)**2 * s_low**2 + w**2 * s_high**2
        s_blend = np.sqrt(var_blend)
        s_blend = np.nan_to_num(s_blend, nan=0.0)

        return s_blend

    def p_plus(E_mu, k=1.0):
        p = np.asarray(p_nom(E_mu))
        s = np.asarray(sigma(E_mu))
        return np.clip(p + k * s, 0.0, 1.0)

    def p_minus(E_mu, k=1.0):
        p = np.asarray(p_nom(E_mu))
        s = np.asarray(sigma(E_mu))
        return np.clip(p - k * s, 0.0, 1.0)

    return p_nom, sigma, p_plus, p_minus

def log_poly(x,a, b, c):
    """
    Log polynomial error function usef for PF uncertainty fits.
    
    Parameters
    ----------
    x: array-like shift values (-10,-5,-3,-1,0,1,3,5,10)
    a: normalization
    b: linearity, left/right asymmetry
    c: curvative, how fast function bends (should be less than 0 for exponential decay distributions)
    all params should be in (-inf, 0]
    """
    return np.exp(a + b*x + c * x**2)


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

import numpy as np

def fit_logpoly_from_dict(pf_dict, pf_floor=1e-12):
    """
    Linear regression using log-polynomial:
        log(PF) = alpha + beta*epsilon + gamma*epsilon^2

    Parameters
    ----------
    pf_dict : dict
        Keys like 'nominal', '+1σ', '-3σ', etc.
    pf_floor : float
        Minimum PF value used to avoid log(0).

    Returns
    -------
    alpha, beta, gamma : floats
    """

    eps_map = {
        'nominal': 0,
        '+1σ': 1,
        '-1σ': -1,
        '+3σ': 3,
        '-3σ': -3,
        '+5σ': 5,
        '-5σ': -5,
        '+10σ': 10,
        '-10σ': -10,
    }

    eps = []
    pf  = []

    for key, epsilon in eps_map.items():
        if key not in pf_dict:
            continue

        value = float(pf_dict[key])

        value = max(value, pf_floor)

        eps.append(epsilon)
        pf.append(value)

    eps = np.asarray(eps, dtype=float)
    pf  = np.asarray(pf, dtype=float)



    X = np.vstack([
        np.ones_like(eps),
        eps,
        eps**2
    ]).T

    y = np.log(pf)

    # Linear regression
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    alpha, beta, gamma = coeffs

    return float(alpha), float(beta), float(gamma)

from scipy.optimize import curve_fit
import numpy as np


def bounded_gompertz(epsilon, a, b, S, C):
    """
    Bounded Gompertz function for passing fraction (PF):

        PF(ε) = (1 - C) * S * exp(-exp(a + b*ε)) + C

    Parameters
    ----------
    epsilon : array-like
        Sigma shift (ε), e.g. 0, ±1, ±3, ±5, ±10.
        Represents the uncertainty variation of the passing fraction.

    a : float
        Horizontal shift of the curve.
        Controls where the transition between high PF and low PF occurs in ε.

    b : float
        Slope parameter (b ≥ 0).
        Controls how quickly PF changes with ε.
        Larger b → sharper transition.

    S : float (0 ≤ S ≤ 1)
        Amplitude scaling factor.
        Controls the maximum excursion above the floor C.
        The maximum PF is:
            PF_max = (1 - C)*S + C

    C : float (0 ≤ C ≤ 1)
        Floor (minimum value) of the passing fraction.
        As ε → +∞:
            PF → C

        Special case:
            C = 1 → PF = 1 for all ε (used for upgoing events)

    Notes
    -----
    - PF is guaranteed to remain within [C, (1 - C)*S + C] ⊆ [0, 1]
    - As ε → -∞:
          PF → (1 - C)*S + C  (maximum)
    - As ε → +∞:
          PF → C              (minimum)
    - Typically monotonic decreasing if b > 0
    """
    epsilon = np.asarray(epsilon, dtype=float)
    A = (1 - C) * S
    return A * np.exp(-np.exp(a + b * epsilon)) + C


import numpy as np
from scipy.optimize import curve_fit


def fit_bounded_gompertz_from_dict(
    pf_dict,
    pf_floor=1e-12,
    const_tol=1e-6,
):
    """
    Fit bounded Gompertz to PF vs epsilon, with safeguards.

    Handles:
    - PF = 1 (upgoing events)
    - Nearly constant PF
    - Numerical failures

    Returns
    -------
    a, b, S, C : floats
    """

    eps_map = {
        'nominal': 0,
        '+1σ': 1,
        '-1σ': -1,
        '+3σ': 3,
        '-3σ': -3,
        '+5σ': 5,
        '-5σ': -5,
        '+10σ': 10,
        '-10σ': -10,
    }

    eps = []
    pf  = []

    for key, e in eps_map.items():
        if key not in pf_dict:
            continue

        val = float(pf_dict[key])
        val = max(val, pf_floor)

        eps.append(e)
        pf.append(val)

    eps = np.asarray(eps, dtype=float)
    pf  = np.asarray(pf, dtype=float)

    #PF = 1
    if np.all(np.isclose(pf, 1.0, atol=const_tol)):
        return 0.0, 0.0, 1.0, 1.0   # C=1 enforces PF=1

    #PF = C
    if np.std(pf) < const_tol:
        C = float(np.mean(pf))
        return 0.0, 0.0, 1.0, C

    #PF normal
    try:
        popt, _ = curve_fit(
            bounded_gompertz,
            eps,
            pf,
            p0=[0.0, 0.1, 0.8, 0.0],
            bounds=(
                [-np.inf, 0, 0, 0],
                [ np.inf, np.inf, 1.0, 1.0]
            ),
            maxfev=10000
        )

        a, b, S, C = popt

    except Exception:
       
        C = float(np.mean(pf))
        a, b, S = 0.0, 0.0, 1.0

    return float(a), float(b), float(S), float(C)