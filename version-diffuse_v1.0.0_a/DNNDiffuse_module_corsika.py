#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT /home/zrechav/i3/icetray/build

"""
DNNDiffuse v1.0.0
- final level cuts, after preferred fit reconstruction
- Writes i3 + HDF output
"""

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, hdfwriter

import argparse
import time

from modules.add_boundaries import make_boundary_check
from modules.add_deposited_energy import add_deposited_energy
from modules.add_snowstorm import map_snowstorm_parameters
from modules.add_PolyplopiaPrimary import add_PolyplopiaPrimary
from modules.add_pf_features import add_depth, add_time, add_cc_tag
from modules.add_DNNDiffuse import DNNDiffuseFinalLevel_v1_0_0_corsika

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input i3 file")
parser.add_argument("-o", "--output", required=True, help="Output file prefix")
parser.add_argument("-g","--gcd", help="GCD file",
 default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz")

args = parser.parse_args()

start_time = time.asctime()
print("Started:", start_time)


tray = I3Tray()

infiles = [args.gcd, args.input]
tray.AddSegment(dataio.I3Reader, "reader", FilenameList=infiles)




tray.Add(make_boundary_check(args.gcd), "boundary_check")
#deposited energy
tray.Add(add_deposited_energy, "store_deposited_energy")

# polyplopia primary
tray.AddModule(add_PolyplopiaPrimary, "add_PolyplopiaPrimary")

#bdt features
tray.Add(add_depth, "addDepth")
tray.Add(add_time, "addTime")
#final level cuts
tray.Add(DNNDiffuseFinalLevel_v1_0_0_corsika,"DNNDiffuseFinalLevel_v1_0_0_corsika")


# i3 output
tray.AddModule(
    "I3Writer",
    "i3_writer",
    filename=args.output + ".i3.zst",
    Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics,icetray.I3Frame.Stream("M")],
    DropOrphanStreams=[icetray.I3Frame.DAQ,icetray.I3Frame.Stream("M")])

# HDF output
tray.Add(
    hdfwriter.I3HDFWriter,
    "hdf_writer",
    Output=args.output + ".hdf5",
    SubEventStreams=["InIceSplit"],
    keys=[
        "DNNDiffuse_v1.0.0_pass",
        "DNNDiffuse_v1.0.0_reco_features",
        "DNNDiffuse_v1.0.0_sigma_features",
        "PreferredFit",
        "contained",
        "partial",
        "Deposited_Energy",
        "MCPrimaryDepth",
        "MCPrimaryTime",
        "I3EventHeader",
        "I3CorsikaInfo",
        "MCPrimary",
        "PolyplopiaInfo",
        "PolyplopiaPrimary",

    ],
)

tray.AddModule("TrashCan", "the_can")
tray.Execute()
tray.Finish()

print("Finished:", time.asctime())
