#pf_bdt.py
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

from icecube import dataclasses
from nuVeto.nuveto import passing
from nuVeto.mu import interp
from nuVeto.utils import Units
import crflux.models as pm

from pf_utils import (
    getDepth,
    getDetectorTime,
    p_light_per_event,
    p_light_per_event_with_uncertainty_blended,
    fit_logpoly_from_dict,
    fit_bounded_gompertz_from_dict,
)
    
def make_AddNewPassingFractions_BDT(
    *,
    bdt_low,
    bdt_high,
    feature_names,
    preach_file,
    density_dict,
    sigma_of_p_low,
    sigma_of_p_high,
    density_key="January",
    hadr="SIBYLL2.3c",
    pmodel=(pm.HillasGaisser2012, "H4a"),
    depth_step_km=0.05,
):

    SIGMA_SHIFTS = [
        ("nominal", 0),
        ("+1σ", +1),
        ("-1σ", -1),
        ("+3σ", +3),
        ("-3σ", -3),
        ("+5σ", +5),
        ("-5σ", -5),
        ("+10σ", +10),
        ("-10σ", -10),
    ]
    
    flav_dict = {
        12: "e",
        -12: "e",
        14: "mu",
        -14: "mu",
        16: "e",
        -16: "e",
    }

    def AddNewPassingFractions_BDT(frame):

        if "DNNDiffuse_v1.0.0_PF_features" not in frame:
            return
        
        if "DNNDiffuse_v1.0.0_pass" not in frame:
            return

        if not frame["DNNDiffuse_v1.0.0_pass"].value:
            return

        cos_zenith = np.round(np.cos(frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_zenith']), 2)

        pfs = dataclasses.I3MapStringDouble()

        # Upgoing → PF = 1
        if cos_zenith < 0:
            for label, _ in SIGMA_SHIFTS:
                pfs[f"PF_conv_{label}"] = 1.0
                pfs[f"PF_pr_{label}"] = 1.0
                
                # LogPoly
                # pfs["PF_conv_logpoly_alpha"] = 0
                # pfs["PF_conv_logpoly_beta"]  = 0
                # pfs["PF_conv_logpoly_gamma"] = 0
                # pfs["PF_pr_logpoly_alpha"] = 0
                # pfs["PF_pr_logpoly_beta"]  = 0
                # pfs["PF_pr_logpoly_gamma"] = 0

                # Gompertz
#                 pfs["PF_conv_gomp_a"] = 0.0
#                 pfs["PF_conv_gomp_b"] = 0.0
#                 pfs["PF_conv_gomp_S"] = 1.0
#                 pfs["PF_conv_gomp_C"] = 1.0

#                 pfs["PF_pr_gomp_a"] = 0.0
#                 pfs["PF_pr_gomp_b"] = 0.0
#                 pfs["PF_pr_gomp_S"] = 1.0
#                 pfs["PF_pr_gomp_C"] = 1.0
                
            frame.Put("AtmNuPassingFraction_BDT", pfs)
            return

        try:
            true_energy = frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_energy']
            true_azimuth = np.degrees(frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_azimuth'])
            true_x = frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_x']
            true_y = frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_y']
            delta_time = frame['DNNDiffuse_v1.0.0_PF_features']['delta_time']
            depo_energy = frame['DNNDiffuse_v1.0.0_PF_features']['deposited_neutrino_energy']
            mbdt = frame['DNNDiffuse_v1.0.0_PF_features']['score_muon_BDT']
            primary_flavor = frame['DNNDiffuse_v1.0.0_PF_features']['flavor']
            cc_tag = frame['DNNDiffuse_v1.0.0_PF_features']['cc_tag']
            depth_at_entry = frame['DNNDiffuse_v1.0.0_PF_features']['depth_at_entry'] #m

            depth_km = np.round(np.round((depth_at_entry / 1000.0) / depth_step_km)
                                * depth_step_km* Units.km,2,)
        except Exception:
            return

        event_dict = {
            "depo_energy": depo_energy,
            "true_zenith": cos_zenith,
            "true_azimuth": true_azimuth,
            "true_x": true_x,
            "true_y": true_y,
            "delta_time": delta_time,
            "muon_BDT": mbdt,
            "primary_depth": depth_at_entry,
            "primary_flavor": primary_flavor,
            "tag": cc_tag,
            "muonbundle_energy": 0.0,
        }

        # 2D array because LightGBM expects (n_samples, n_features)
        event_array = np.array(
            [[event_dict[f] for f in feature_names]],
            dtype=float,
        )

        # If p_light_per_event_with_uncertainty_blended expects 1D row:
        event_row = event_array[0]

        ##KFold CV
        p_nom, p_sig, p_plus, p_minus = p_light_per_event_with_uncertainty_blended(
            event_row=event_row,
            bdt_low=bdt_low,
            bdt_high=bdt_high,
            feature_names=feature_names,
            sigma_of_p_low=sigma_of_p_low,
            sigma_of_p_high=sigma_of_p_high,
        )

        flav = flav_dict[int(primary_flavor)]

        for label, k in SIGMA_SHIFTS:

            if k == 0:
                p_light_evt = p_nom
            elif k > 0:
                p_light_evt = lambda E_mu, kk=k: p_plus(E_mu, kk)
            else:
                p_light_evt = lambda E_mu, kk=abs(k): p_minus(E_mu, kk)

            prpl = interp(preach_file, p_light_evt)

            kind_conv = f"conv nu_{flav}"
            kind_pr   = f"pr nu_{flav}"

            PF_conv = passing(
                true_energy,
                cos_zenith,
                kind=kind_conv,
                pmodel=pmodel,
                hadr=hadr,
                depth=depth_km,
                density=("CORSIKA", density_dict[density_key]),
                prpl=prpl,
            )

            PF_pr = passing(
                true_energy,
                cos_zenith,
                kind=kind_pr,
                pmodel=pmodel,
                hadr=hadr,
                depth=depth_km,
                density=("CORSIKA", density_dict[density_key]),
                prpl=prpl,
            )

            pfs[f"PF_conv_{label}"] = float(PF_conv)
            pfs[f"PF_pr_{label}"] = float(PF_pr)
            
        pf_conv_dict = {label: pfs[f"PF_conv_{label}"] for label, _ in SIGMA_SHIFTS}
        pf_pr_dict   = {label: pfs[f"PF_pr_{label}"]   for label, _ in SIGMA_SHIFTS}

        # logpoly
        # a_c, b_c, g_c = fit_logpoly_from_dict(pf_conv_dict)
        # a_p, b_p, g_p = fit_logpoly_from_dict(pf_pr_dict)

        # gompert
        #a_c, b_c, S_c, C_c = fit_bounded_gompertz_from_dict(pf_conv_dict)
        #a_p, b_p, S_p, C_p = fit_bounded_gompertz_from_dict(pf_pr_dict)

        # logpoly
        # pfs["PF_conv_logpoly_alpha"] = a_c
        # pfs["PF_conv_logpoly_beta"]  = b_c
        # pfs["PF_conv_logpoly_gamma"] = g_c
        # pfs["PF_pr_logpoly_alpha"] = a_p
        # pfs["PF_pr_logpoly_beta"]  = b_p
        # pfs["PF_pr_logpoly_gamma"] = g_p

        # gompert
#         pfs["PF_conv_gomp_a"] = a_c
#         pfs["PF_conv_gomp_b"] = b_c
#         pfs["PF_conv_gomp_S"] = S_c
#         pfs["PF_conv_gomp_C"] = C_c

#         pfs["PF_pr_gomp_a"] = a_p
#         pfs["PF_pr_gomp_b"] = b_p
#         pfs["PF_pr_gomp_S"] = S_p
#         pfs["PF_pr_gomp_C"] = C_p

        frame.Put("AtmNuPassingFraction_BDT", pfs)

    return AddNewPassingFractions_BDT

    
from icecube import icetray, dataclasses
import numpy as np

# make sure this is imported from your pf_utils
# from pf_utils import fit_bounded_gompertz_from_dict


def AddGompertzFits(frame):

    SIGMA_LABELS = [
        "nominal",
        "+1σ", "-1σ",
        "+3σ", "-3σ",
        "+5σ", "-5σ",
        "+10σ", "-10σ",
    ]

    if "AtmNuPassingFraction_BDT" not in frame:
        return

    pfs = frame["AtmNuPassingFraction_BDT"]

    pf_conv_dict = {}
    pf_pr_dict   = {}

    for label in SIGMA_LABELS:
        key_c = f"PF_conv_{label}"
        key_p = f"PF_pr_{label}"

        if key_c in pfs:
            pf_conv_dict[label] = pfs[key_c]

        if key_p in pfs:
            pf_pr_dict[label] = pfs[key_p]

    if len(pf_conv_dict) < 3 or len(pf_pr_dict) < 3:
        return

    try:
        a_c, b_c, S_c, C_c = fit_bounded_gompertz_from_dict(pf_conv_dict)
        a_p, b_p, S_p, C_p = fit_bounded_gompertz_from_dict(pf_pr_dict)

    except Exception:
        a_c = b_c = 0.0
        S_c = C_c = 1.0
        a_p = b_p = 0.0
        S_p = C_p = 1.0

    pfs["PF_conv_gomp_a"] = float(a_c)
    pfs["PF_conv_gomp_b"] = float(b_c)
    pfs["PF_conv_gomp_S"] = float(S_c)
    pfs["PF_conv_gomp_C"] = float(C_c)

    pfs["PF_pr_gomp_a"] = float(a_p)
    pfs["PF_pr_gomp_b"] = float(b_p)
    pfs["PF_pr_gomp_S"] = float(S_p)
    pfs["PF_pr_gomp_C"] = float(C_p)

    frame.Put("AtmNuPassingFraction_BDT_Gompert_Params", pfs)