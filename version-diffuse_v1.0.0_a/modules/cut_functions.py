from icecube import dataclasses

from versions.DNNDiffuse_v1_0_0 import (ENERGY_CONTAINED_Z,CASCADE_BDT_MIN, MUON_BDT_MAX,CONTAINED_QTOT_MIN,PARTIAL_QTOT_MIN,THEO_SCORE_MIN)




def z_energy_bottom_slice(frame, threshold=ENERGY_CONTAINED_Z):
    """
    Reject if z in [-700, -500] AND energy < threshold
    """
    flag = True
    if "PreferredFit" in frame:
        p = frame["PreferredFit"]
        z = p.pos.z
        energy = p.energy

        if -700.0 <= z <= -500.0 and energy < threshold:
            flag = False

    return flag


def z_energy_uncontained(frame, threshold=ENERGY_CONTAINED_Z):
    """
    Reject if energy < threshold AND event is uncontained (partial)
    """
    flag = True
    if "PreferredFit" in frame:
        energy = frame["PreferredFit"].energy
        partial = frame['partial'].value

        if energy < threshold and partial:
            flag = False

    return flag




def cascade_BDT_cut(frame, threshold=CASCADE_BDT_MIN):
    flag = False
    if "PreferredFit" in frame:
        val = frame[
            "BDT_bdt_max_depth_4_n_est_1000lr_0.01_seed_3_train_size_50"
        ]["pred_001"]
        if val >= threshold:
            flag = True
    return flag


def muon_BDT_cut(frame, threshold=MUON_BDT_MAX):
    flag = False
    if "PreferredFit" in frame:
        val = frame[
            "BDT_bdt_max_depth_4_n_est_2000lr_0_02_seed_3_train_size_50"
        ]["pred_001"]
        if val < threshold:
            flag = True
    return flag

def qtot_cut(frame, thresholdc=CONTAINED_QTOT_MIN,thresholdp=PARTIAL_QTOT_MIN):
    flag = False
    if "PreferredFit" in frame:
        if frame['contained'].value == True:
            val = frame["Homogenized_QTot"].value
            if val > thresholdc:
                flag = True
        if frame['partial'].value == True:
            val = frame["Homogenized_QTot"].value
            if val > thresholdp:
                flag = True
    return flag
    
def theo_cut(frame, threshold=THEO_SCORE_MIN):
    flag = False
    if "PreferredFit" in frame:
        val = frame['EventClassifierOutput']['Through_Going_Track']
        if val < threshold:
            flag = True
    return flag