from icecube import dataclasses
from icecube import icetray
from icecube.dataclasses import I3MapStringDouble

from versions.DNNDiffuse_v1_0_0 import (
    # geometry / energy
    Z_LOWER,
    Z_UPPER,
    DUST_MIN,
    DUST_MAX,
    MIN_RECO_ENERGY,

    # bools
    USE_DUST_CUT,
    USE_BDT_CUTS,
    USE_ENERGY_Z_CUT,
    #energy dependent geometry cut method
    ENERGY_Z_MODE,   # options: "bottom_slice" or "uncontained"
)


from modules.cut_functions import (
    z_energy_bottom_slice,
    z_energy_uncontained,
    cascade_BDT_cut,
    muon_BDT_cut,
)


def DNNDiffuseFinalLevel_v1_0_0(frame):
    """
    Apply DNNDiffuse v1.0.0 final-level cuts.

    Always adds:
        DNNDiffuse_v1.0.0_pass
        DNNDiffuse_v1.0.0_reco_features
        DNNDiffuse_v1.0.0_PF_features (to be filled later)
    """

    passes = True

    reco_vars = I3MapStringDouble()
    # pf_features = I3MapStringDouble()  # fill later



    if "PreferredFit" not in frame:
        passes = False
    else:
        p = frame["PreferredFit"]
        
        x = p.pos.x
        y = p.pos.y
        z = p.pos.z
        zenith = p.dir.zenith
        azimuth = p.dir.azimuth
        energy = p.energy
        
        reco_vars["reco_x"] = x
        reco_vars["reco_y"] = y
        reco_vars["reco_z"] = z
        reco_vars["reco_zenith"] = zenith
        reco_vars["reco_azimuth"] = azimuth
        reco_vars["reco_energy"] = energy

        contained = frame['contained'].value
        partial   = frame['partial'].value

        reco_vars["contained"] = contained
        reco_vars["partial"]   = partial

        reco_vars["score_cascade_BDT"] = frame["BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50"
                                        ]["pred_001"]
        reco_vars["score_muon_BDT"] = frame["BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50"
                                        ]["pred_001"]
        if energy < MIN_RECO_ENERGY:
            passes = False

        if not (contained or partial):
            passes = False

        if not (Z_LOWER <= z <= Z_UPPER):
            passes = False

        # energy dependent geometry cuts
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

    frame["DNNDiffuse_v1.0.0_pass"] = icetray.I3Bool(passes)
    frame["DNNDiffuse_v1.0.0_reco_features"] = reco_vars
    # frame["DNNDiffuse_v1.0.0_PF_features"] = pf_features

    return True

