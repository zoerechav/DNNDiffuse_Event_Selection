#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT /data/user/zrechav/i3/icetray/build/

import argparse
import lightgbm as lgb
import pickle

from icecube.icetray import I3Tray
from icecube import icetray

import crflux.models as pm

from pf_bdt import make_AddNewPassingFractions_BDT, AddGompertzFits
from pf_table import make_AddNewPassingFractions_Table
from pf_utils import build_sigma_of_p_from_cv


DEFAULT_GCD = (
    "/cvmfs/icecube.opensciencegrid.org/data/GCD/""GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz")

parser = argparse.ArgumentParser()

parser.add_argument("--infile", required=True, help="Input I3 file")
parser.add_argument("--outfile", required=True, help="Output I3 file")
parser.add_argument("--gcd", default=DEFAULT_GCD, help="GCD file")

args = parser.parse_args()


# --------------------------------------------------
# BDT PATHS (NEW BLENDED MODEL)
# --------------------------------------------------

LOW_BDT_PATH = (
    "/data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/bdt_uncertainty/bdts/"
    "REPEATED_KFOLD_lPeV_depo_energy/NOMINAL_BDT/bdt_model.txt"
)

HIGH_BDT_PATH = (
    "/data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/bdt_uncertainty/bdts/"
    "REPEATED_KFOLD_20260206_082552/NOMINAL_BDT/bdt_model.txt"
)

LOW_CV_PATH = (
    "/data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/bdt_uncertainty/bdts/"
    "REPEATED_KFOLD_lPeV_depo_energy/NOMINAL_BDT/nominal_predictions_only.pkl"
)

HIGH_CV_PATH = (
    "/data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/bdt_uncertainty/bdts/"
    "REPEATED_KFOLD_20260206_082552/NOMINAL_BDT/nominal_predictions_only.pkl"
)

PREACH_FILE = (
    "/data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/build_PF/npx/ice_allm97.npz"
)

density_dict = {
    "January":  ("PL_SouthPole", "January"),
    "June":     ("SouthPole", "June"),
    "August":   ("PL_SouthPole", "August"),
    "December": ("SouthPole", "December"),
}

PF_FOLDER = "/data/user/zrechav/calc_PF/PFs_v7"


print("load bdt models")

bdt_low  = lgb.Booster(model_file=LOW_BDT_PATH)
bdt_high = lgb.Booster(model_file=HIGH_BDT_PATH)

feature_names = bdt_low.feature_name()

sigma_of_p_low  = build_sigma_of_p_from_cv(LOW_CV_PATH)
sigma_of_p_high = build_sigma_of_p_from_cv(HIGH_CV_PATH)

print('loaded')


AddPF_BDT = make_AddNewPassingFractions_BDT(
    bdt_low=bdt_low,
    bdt_high=bdt_high,
    feature_names=feature_names,
    preach_file=PREACH_FILE,
    density_dict=density_dict,
    sigma_of_p_low=sigma_of_p_low,
    sigma_of_p_high=sigma_of_p_high,
    density_key="January",
    pmodel=(pm.HillasGaisser2012, "H4a"),
    hadr="SIBYLL2.3c",
    depth_step_km=0.05,
)

AddPF_Table = make_AddNewPassingFractions_Table(
    pf_folder=PF_FOLDER,
    depth_space=(1.4, 2.1),
    shifts=(-3, -1, 0, 3, 10),
)

# --------------------------------------------------
# I3Tray
# --------------------------------------------------

tray = I3Tray()

tray.AddModule(
    "I3Reader",
    "reader",
    filenamelist=[args.gcd, args.infile],
)

def keep_passed(frame):

    if "DNNDiffuse_v1.0.0_pass" not in frame:
        return False

    return frame["DNNDiffuse_v1.0.0_pass"].value

tray.Add(keep_passed, "keep_passed", Streams=[icetray.I3Frame.Physics])


# Table-based PF (fast)
tray.AddModule(AddPF_Table, "AddPassingFractionsTable")

# New blended-BDT PF (slow)
tray.AddModule(AddPF_BDT, "AddPassingFractionsBDT")

#tray.AddModule(AddGompertzFits, "AddGompertzFits")

tray.AddModule(
    "I3Writer",
    "writer",
    filename=args.outfile + ".i3.zst",
    Streams=[
        icetray.I3Frame.TrayInfo,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics,
    ],
    DropOrphanStreams=[icetray.I3Frame.DAQ],
)

tray.AddModule("TrashCan", "trash")

tray.Execute()
tray.Finish()