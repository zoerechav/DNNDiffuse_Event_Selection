from icecube import dataclasses, millipede
from icecube import icetray
from icecube.dataclasses import I3MapStringDouble
import numpy as np

from versions.DNNDiffuse_v1_0_0 import (
    # geometry / energy
    Z_LOWER,
    Z_UPPER,
    DUST_MIN,
    DUST_MAX,
    MIN_RECO_ENERGY,
    CONTAINED_QTOT_MIN,
    PARTIAL_QTOT_MIN,
    THEO_SCORE_MIN,

    # bools
    USE_DUST_CUT,
    USE_BDT_CUTS,
    USE_ENERGY_Z_CUT,
    USE_QTOT_CUTS,
    USE_THEO_CUTS,
    #energy dependent geometry cut method
    ENERGY_Z_MODE,   # options: "bottom_slice" or "uncontained"
    
)


from modules.cut_functions import (
    z_energy_bottom_slice,
    z_energy_uncontained,
    cascade_BDT_cut,
    muon_BDT_cut,
    qtot_cut,
    theo_cut,
)

from utils.pf_features import getDepth, getDetectorTime

from utils.sigma_uncertainty_features import dom_location,convex_hull, GetDistance,  find_shortest_distance, opening_angle

def DNNDiffuseFinalLevel_v1_0_0_nugen(frame):
    """
    Apply DNNDiffuse v1.0.0 final-level cuts.

    Always adds:
        DNNDiffuse_v1.0.0_pass
        DNNDiffuse_v1.0.0_reco_features
        DNNDiffuse_v1.0.0_PF_features
        DNNDiffuse_v1.0.0_sigma_features
    """

    passes = True

    reco_vars = I3MapStringDouble()
    pf_vars = I3MapStringDouble()
    sigma_vars = I3MapStringDouble()

    if "PreferredFit" not in frame:
        passes = False
    else:
        p = frame["PreferredFit"]

        #reco features
        x = p.pos.x
        y = p.pos.y
        z = p.pos.z
        zenith = p.dir.zenith
        azimuth = p.dir.azimuth
        energy = p.energy
        length = getattr(p, "length", 0.0)

        reco_vars["reco_x"] = x
        reco_vars["reco_y"] = y
        reco_vars["reco_z"] = z
        reco_vars["reco_zenith"] = zenith
        reco_vars["reco_azimuth"] = azimuth
        reco_vars["reco_energy"] = energy
        #reco_vars["length"] = length

        contained = frame["contained"].value
        partial = frame["partial"].value

        reco_vars["contained"] = contained
        reco_vars["partial"] = partial

        cascade_score = frame[
            "BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50"
        ]["pred_001"]

        muon_score = frame[
            "BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50"
        ]["pred_001"]

        reco_vars["score_cascade_BDT"] = cascade_score
        reco_vars["score_muon_BDT"] = muon_score
        
        reco_vars["homogenized_qtot"] = frame['Homogenized_QTot'].value
        
        reco_vars["throughgoing_score"] = frame['EventClasssifierOutput']['Through_Going_Track']

        #cuts
        if energy < MIN_RECO_ENERGY:
            passes = False

        if not (contained or partial):
            passes = False

        if not (Z_LOWER <= z <= Z_UPPER):
            passes = False

        if USE_ENERGY_Z_CUT:
            if ENERGY_Z_MODE == "bottom_slice":
                if not z_energy_bottom_slice(frame):
                    passes = False
            elif ENERGY_Z_MODE == "uncontained":
                if not z_energy_uncontained(frame):
                    passes = False

        if USE_DUST_CUT:
            if DUST_MIN < z < DUST_MAX:
                passes = False

        if USE_BDT_CUTS:
            if not cascade_BDT_cut(frame):
                passes = False
            if not muon_BDT_cut(frame):
                passes = False
                
        if USE_QTOT_CUTS:
            if not qtot_cut(frame):
                passes = False

        if USE_THEO_CUTS:
            if not thoe_cut(frame):
                passes = False
            

        #pf features
        cc = frame["cc"]

        tru_x = cc.pos.x
        tru_y = cc.pos.y
        tru_zenith = cc.dir.zenith
        tru_azimuth = cc.dir.azimuth
        tru_energy = cc.energy
        depo_energy = frame["Deposited_Energy"].value

        tru_t = cc.time
        bundle_t = frame["MCPrimaryTime"].value
        delta_t = tru_t - bundle_t

        depth_at_entry = frame["MCPrimaryDepth"].value
        cc_tag = frame["MC_CCTag"].value
        flavor = frame["PolyplopiaPrimary"].type

        pf_vars["deposited_neutrino_energy"] = depo_energy
        pf_vars["true_neutrino_energy"] = tru_energy
        pf_vars["true_neutrino_x"] = tru_x
        pf_vars["true_neutrino_y"] = tru_y
        pf_vars["true_neutrino_zenith"] = tru_zenith
        pf_vars["true_neutrino_azimuth"] = tru_azimuth
        pf_vars["depth_at_entry"] = depth_at_entry
        pf_vars["delta_time"] = delta_t
        pf_vars["flavor"] = flavor
        pf_vars["cc_tag"] = cc_tag
        pf_vars["score_muon_BDT"] = muon_score

        #sigma uncertainty parameters
        rlogl = 0.0
        q_tot = 0.0
        ndof = 0.0
        chisqr = 0.0
        sqrresi = 0.0

        fit_key = frame["PreferredFit_key"]

        if fit_key == "TaupedeFit_iMIGRAD_PPB0":
            params = frame["TaupedeFit_iMIGRAD_PPB0FitParams"]
        elif fit_key == "MonopodFit_iMIGRAD_PPB0":
            params = frame["MonopodFit_iMIGRAD_PPB0FitParams"]
        else:
            params = None

        if params is not None:
            rlogl = params.rlogl
            q_tot = params.qtotal
            ndof = params.ndof
            chisqr = params.chi_squared
            sqrresi = params.squared_residuals

        
        dom_loc = dom_location()
        points = np.array([dom_loc["x"], dom_loc["y"]]).T
        hull = convex_hull(points)

        
        dist_detector_edge = GetDistance(
            x,
            y,
            hull[:, 0],
            hull[:, 1],
        )

        
        dist_to_string = find_shortest_distance(dom_loc, x, y)

        
        oa = opening_angle(
            zenith,
            azimuth,
            tru_zenith,
            tru_azimuth,
        )

        
        sigma_vars["reco_x"] = x
        sigma_vars["reco_y"] = y
        sigma_vars["reco_z"] = z
        sigma_vars["reco_azi"] = azimuth
        sigma_vars["reco_zen"] = zenith
        sigma_vars["reco_energy"] = energy
        sigma_vars["length"] = length

        sigma_vars["true_azi"] = tru_azimuth
        sigma_vars["true_zen"] = tru_zenith
        sigma_vars["true_energy"] = tru_energy

        sigma_vars["q_tot"] = q_tot
        sigma_vars["ndof"] = ndof
        sigma_vars["rlogl"] = rlogl
        sigma_vars["sqrresi"] = sqrresi
        sigma_vars["chisqr"] = chisqr

        sigma_vars["oangle"] = oa
        sigma_vars["detector_edge"] = dist_detector_edge
        sigma_vars["dist_to_string"] = dist_to_string

    # ----------------------------------------
    # Attach to frame
    # ----------------------------------------
    frame["DNNDiffuse_v1.0.0_pass"] = icetray.I3Bool(passes)
    frame["DNNDiffuse_v1.0.0_reco_features"] = reco_vars
    frame["DNNDiffuse_v1.0.0_PF_features"] = pf_vars
    frame["DNNDiffuse_v1.0.0_sigma_features"] = sigma_vars

    return True

def DNNDiffuseFinalLevel_v1_0_0_corsika(frame):

    passes = True

    reco_vars = I3MapStringDouble()
    sigma_vars = I3MapStringDouble()

    if "PreferredFit" not in frame:
        passes = False
    else:
        p = frame["PreferredFit"]

        #reco features
        x = p.pos.x
        y = p.pos.y
        z = p.pos.z
        zenith = p.dir.zenith
        azimuth = p.dir.azimuth
        energy = p.energy
        length = getattr(p, "length", 0.0)

        reco_vars["reco_x"] = x
        reco_vars["reco_y"] = y
        reco_vars["reco_z"] = z
        reco_vars["reco_zenith"] = zenith
        reco_vars["reco_azimuth"] = azimuth
        reco_vars["reco_energy"] = energy
        #reco_vars["length"] = length

        contained = frame["contained"].value
        partial = frame["partial"].value

        reco_vars["contained"] = contained
        reco_vars["partial"] = partial

        cascade_score = frame[
            "BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50"
        ]["pred_001"]

        muon_score = frame[
            "BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50"
        ]["pred_001"]

        reco_vars["score_cascade_BDT"] = cascade_score
        reco_vars["score_muon_BDT"] = muon_score
        
        reco_vars["homogenized_qtot"] = frame['Homogenized_QTot'].value
        
        reco_vars["throughgoing_score"] = frame['EventClasssifierOutput']['Through_Going_Track']

        #cuts
        if energy < MIN_RECO_ENERGY:
            passes = False

        if not (contained or partial):
            passes = False

        if not (Z_LOWER <= z <= Z_UPPER):
            passes = False

        if USE_ENERGY_Z_CUT:
            if ENERGY_Z_MODE == "bottom_slice":
                if not z_energy_bottom_slice(frame):
                    passes = False
            elif ENERGY_Z_MODE == "uncontained":
                if not z_energy_uncontained(frame):
                    passes = False

        if USE_DUST_CUT:
            if DUST_MIN < z < DUST_MAX:
                passes = False

        if USE_BDT_CUTS:
            if not cascade_BDT_cut(frame):
                passes = False
            if not muon_BDT_cut(frame):
                passes = False
                
        if USE_QTOT_CUTS:
            if not qtot_cut(frame):
                passes = False

        if USE_THEO_CUTS:
            if not thoe_cut(frame):
                passes = False

        #sigma uncertainty features

        cc = frame["cc"]

        true_zen = cc.dir.zenith
        true_azi = cc.dir.azimuth
        true_energy = cc.energy

        
        oa = opening_angle(
            zenith,
            azimuth,
            true_zen,
            true_azi,
        )

        
        rlogl = 0.0
        q_tot = 0.0
        ndof = 0.0
        chisqr = 0.0
        sqrresi = 0.0

        fit_key = frame["PreferredFit_key"]

        if fit_key == "TaupedeFit_iMIGRAD_PPB0":
            params = frame["TaupedeFit_iMIGRAD_PPB0FitParams"]
        elif fit_key == "MonopodFit_iMIGRAD_PPB0":
            params = frame["MonopodFit_iMIGRAD_PPB0FitParams"]
        else:
            params = None

        if params is not None:
            rlogl = params.rlogl
            q_tot = params.qtotal
            ndof = params.ndof
            chisqr = params.chi_squared
            sqrresi = params.squared_residuals

        
        dom_loc = dom_location()
        points = np.array([dom_loc["x"], dom_loc["y"]]).T
        hull = convex_hull(points)

        dist_detector_edge = GetDistance(
            x,
            y,
            hull[:, 0],
            hull[:, 1],
        )

        dist_to_string = find_shortest_distance(dom_loc, x, y)

        
        sigma_vars["reco_x"] = x
        sigma_vars["reco_y"] = y
        sigma_vars["reco_z"] = z
        sigma_vars["reco_azi"] = azimuth
        sigma_vars["reco_zen"] = zenith
        sigma_vars["reco_energy"] = energy
        sigma_vars["length"] = length

        sigma_vars["true_azi"] = true_azi
        sigma_vars["true_zen"] = true_zen
        sigma_vars["true_energy"] = true_energy

        sigma_vars["q_tot"] = q_tot
        sigma_vars["ndof"] = ndof
        sigma_vars["rlogl"] = rlogl
        sigma_vars["sqrresi"] = sqrresi
        sigma_vars["chisqr"] = chisqr

        sigma_vars["oangle"] = oa
        sigma_vars["detector_edge"] = dist_detector_edge
        sigma_vars["dist_to_string"] = dist_to_string

    frame["DNNDiffuse_v1.0.0_pass"] = icetray.I3Bool(passes)
    frame["DNNDiffuse_v1.0.0_reco_features"] = reco_vars
    frame["DNNDiffuse_v1.0.0_sigma_features"] = sigma_vars

    return True

def DNNDiffuseFinalLevel_v1_0_0_muongun(frame):

    passes = True

    reco_vars = I3MapStringDouble()
    sigma_vars = I3MapStringDouble()

    if "PreferredFit" not in frame:
        passes = False
    else:
        p = frame["PreferredFit"]

        #reco features
        x = p.pos.x
        y = p.pos.y
        z = p.pos.z
        zenith = p.dir.zenith
        azimuth = p.dir.azimuth
        energy = p.energy
        length = getattr(p, "length", 0.0)

        reco_vars["reco_x"] = x
        reco_vars["reco_y"] = y
        reco_vars["reco_z"] = z
        reco_vars["reco_zenith"] = zenith
        reco_vars["reco_azimuth"] = azimuth
        reco_vars["reco_energy"] = energy
        #reco_vars["length"] = length

        contained = frame["contained"].value
        partial = frame["partial"].value

        reco_vars["contained"] = contained
        reco_vars["partial"] = partial

        cascade_score = frame[
            "BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50"
        ]["pred_001"]

        muon_score = frame[
            "BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50"
        ]["pred_001"]

        reco_vars["score_cascade_BDT"] = cascade_score
        reco_vars["score_muon_BDT"] = muon_score
        
        reco_vars["homogenized_qtot"] = frame['Homogenized_QTot'].value
        
        reco_vars["throughgoing_score"] = frame['EventClasssifierOutput']['Through_Going_Track']

        #cuts
        if energy < MIN_RECO_ENERGY:
            passes = False

        if not (contained or partial):
            passes = False

        if not (Z_LOWER <= z <= Z_UPPER):
            passes = False

        if USE_ENERGY_Z_CUT:
            if ENERGY_Z_MODE == "bottom_slice":
                if not z_energy_bottom_slice(frame):
                    passes = False
            elif ENERGY_Z_MODE == "uncontained":
                if not z_energy_uncontained(frame):
                    passes = False

        if USE_DUST_CUT:
            if DUST_MIN < z < DUST_MAX:
                passes = False

        if USE_BDT_CUTS:
            if not cascade_BDT_cut(frame):
                passes = False
            if not muon_BDT_cut(frame):
                passes = False
        
        if USE_QTOT_CUTS:
            if not qtot_cut(frame):
                passes = False

        if USE_THEO_CUTS:
            if not thoe_cut(frame):
                passes = False

        cc = frame["cc"]

        true_zen = cc.dir.zenith
        true_azi = cc.dir.azimuth
        true_energy = cc.energy

        
        oa = opening_angle(
            zenith,
            azimuth,
            true_zen,
            true_azi,
        )

        
        rlogl = 0.0
        q_tot = 0.0
        ndof = 0.0
        chisqr = 0.0
        sqrresi = 0.0

        fit_key = frame["PreferredFit_key"]

        if fit_key == "TaupedeFit_iMIGRAD_PPB0":
            params = frame["TaupedeFit_iMIGRAD_PPB0FitParams"]
        elif fit_key == "MonopodFit_iMIGRAD_PPB0":
            params = frame["MonopodFit_iMIGRAD_PPB0FitParams"]
        else:
            params = None

        if params is not None:
            rlogl = params.rlogl
            q_tot = params.qtotal
            ndof = params.ndof
            chisqr = params.chi_squared
            sqrresi = params.squared_residuals

        
        dom_loc = dom_location()
        points = np.array([dom_loc["x"], dom_loc["y"]]).T
        hull = convex_hull(points)

        dist_detector_edge = GetDistance(
            x,
            y,
            hull[:, 0],
            hull[:, 1],
        )

        dist_to_string = find_shortest_distance(dom_loc, x, y)

        
        sigma_vars["reco_x"] = x
        sigma_vars["reco_y"] = y
        sigma_vars["reco_z"] = z
        sigma_vars["reco_azi"] = azimuth
        sigma_vars["reco_zen"] = zenith
        sigma_vars["reco_energy"] = energy
        sigma_vars["length"] = length

        sigma_vars["true_azi"] = true_azi
        sigma_vars["true_zen"] = true_zen
        sigma_vars["true_energy"] = true_energy

        sigma_vars["q_tot"] = q_tot
        sigma_vars["ndof"] = ndof
        sigma_vars["rlogl"] = rlogl
        sigma_vars["sqrresi"] = sqrresi
        sigma_vars["chisqr"] = chisqr

        sigma_vars["oangle"] = oa
        sigma_vars["detector_edge"] = dist_detector_edge
        sigma_vars["dist_to_string"] = dist_to_string

    frame["DNNDiffuse_v1.0.0_pass"] = icetray.I3Bool(passes)
    frame["DNNDiffuse_v1.0.0_reco_features"] = reco_vars
    frame["DNNDiffuse_v1.0.0_sigma_features"] = sigma_vars

    return True


def DNNDiffuseFinalLevel_v1_0_0_exp(frame):

    passes = True

    reco_vars = I3MapStringDouble()
    sigma_vars = I3MapStringDouble()

    if "PreferredFit" not in frame:
        passes = False
    else:
        p = frame["PreferredFit"]

        #reco features
        x = p.pos.x
        y = p.pos.y
        z = p.pos.z
        zenith = p.dir.zenith
        azimuth = p.dir.azimuth
        energy = p.energy
        length = getattr(p, "length", 0.0)

        reco_vars["reco_x"] = x
        reco_vars["reco_y"] = y
        reco_vars["reco_z"] = z
        reco_vars["reco_zenith"] = zenith
        reco_vars["reco_azimuth"] = azimuth
        reco_vars["reco_energy"] = energy
        #reco_vars["length"] = length

        contained = frame["contained"].value
        partial = frame["partial"].value

        reco_vars["contained"] = contained
        reco_vars["partial"] = partial

        cascade_score = frame[
            "BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50"
        ]["pred_001"]

        muon_score = frame[
            "BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50"
        ]["pred_001"]

        reco_vars["score_cascade_BDT"] = cascade_score
        reco_vars["score_muon_BDT"] = muon_score
        
        reco_vars["homogenized_qtot"] = frame['Homogenized_QTot'].value
        
        reco_vars["throughgoing_score"] = frame['EventClasssifierOutput']['Through_Going_Track']

        #cuts
        if energy < MIN_RECO_ENERGY:
            passes = False

        if not (contained or partial):
            passes = False

        if not (Z_LOWER <= z <= Z_UPPER):
            passes = False

        if USE_ENERGY_Z_CUT:
            if ENERGY_Z_MODE == "bottom_slice":
                if not z_energy_bottom_slice(frame):
                    passes = False
            elif ENERGY_Z_MODE == "uncontained":
                if not z_energy_uncontained(frame):
                    passes = False

        if USE_DUST_CUT:
            if DUST_MIN < z < DUST_MAX:
                passes = False

        if USE_BDT_CUTS:
            if not cascade_BDT_cut(frame):
                passes = False
            if not muon_BDT_cut(frame):
                passes = False
        
        if USE_QTOT_CUTS:
            if not qtot_cut(frame):
                passes = False

        if USE_THEO_CUTS:
            if not thoe_cut(frame):
                passes = False

        
        rlogl = 0.0
        q_tot = 0.0
        ndof = 0.0
        chisqr = 0.0
        sqrresi = 0.0

        fit_key = frame["PreferredFit_key"]

        if fit_key == "TaupedeFit_iMIGRAD_PPB0":
            params = frame["TaupedeFit_iMIGRAD_PPB0FitParams"]
        elif fit_key == "MonopodFit_iMIGRAD_PPB0":
            params = frame["MonopodFit_iMIGRAD_PPB0FitParams"]
        else:
            params = None

        if params is not None:
            rlogl = params.rlogl
            q_tot = params.qtotal
            ndof = params.ndof
            chisqr = params.chi_squared
            sqrresi = params.squared_residuals

        
        dom_loc = dom_location()
        points = np.array([dom_loc["x"], dom_loc["y"]]).T
        hull = convex_hull(points)

        dist_detector_edge = GetDistance(
            x,
            y,
            hull[:, 0],
            hull[:, 1],
        )

        dist_to_string = find_shortest_distance(dom_loc, x, y)

        
        sigma_vars["reco_x"] = x
        sigma_vars["reco_y"] = y
        sigma_vars["reco_z"] = z
        sigma_vars["reco_azi"] = azimuth
        sigma_vars["reco_zen"] = zenith
        sigma_vars["reco_energy"] = energy
        sigma_vars["length"] = length

        sigma_vars["q_tot"] = q_tot
        sigma_vars["ndof"] = ndof
        sigma_vars["rlogl"] = rlogl
        sigma_vars["sqrresi"] = sqrresi
        sigma_vars["chisqr"] = chisqr

        sigma_vars["detector_edge"] = dist_detector_edge
        sigma_vars["dist_to_string"] = dist_to_string

    frame["DNNDiffuse_v1.0.0_pass"] = icetray.I3Bool(passes)
    frame["DNNDiffuse_v1.0.0_reco_features"] = reco_vars
    frame["DNNDiffuse_v1.0.0_sigma_features"] = sigma_vars

    return True
