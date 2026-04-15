#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/RHEL_9_x86_64_v2/metaprojects/icetray/v1.17.0/bin/icetray-shell

"""
Run Theo's DNN classifier on a single i3 file.
Only create a new output file if one does not already exist.
"""

import numpy as np
from icecube import dataclasses, icetray, linefit, photonics_service, phys_services, dataio
from icecube.phys_services.which_split import which_split
from icecube.offline_filterscripts.base_segments.event_classifier import event_classifier_onnx_ml
from I3Tray import *

from pathlib import Path
import os
import argparse
import time


parser = argparse.ArgumentParser()

parser.add_argument(
    "--infile",
    required=True,
    help="Input i3 file"
)

parser.add_argument(
    "--outfile",
    required=True,
    help="Output i3 file"
)

parser.add_argument(
    "-g", "--gcd",
    default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz",
    help="GCD file"
)

args = parser.parse_args()

start_time = time.asctime()
print("Started:", start_time)

onnx_model_path = "/cvmfs/icecube.opensciencegrid.org/users/briedel/ml/models/tglauch_classifier/3/model.onnx"
ml_config_path = Path(os.environ["I3_BUILD"]) / "ml_suite/resources/theo_dnn_classification_model.yaml"

infile = Path(args.infile)
outfile = Path(args.outfile)

# make parent output directory if needed
outfile.parent.mkdir(parents=True, exist_ok=True)

if outfile.exists():
    print(f"Skipping: output already exists: {outfile}")
    print("Finished:", time.asctime())
    raise SystemExit(0)

print(f"Processing: {infile}")
print(f"Output:     {outfile}")

tray = I3Tray()

infiles = [args.gcd, str(infile)]

tray.Add("I3Reader", "reader", FilenameList=infiles)

tray.AddSegment(
    event_classifier_onnx_ml,
    "EventClassifier",
    ONNX_model_path=str(onnx_model_path),
    mlsuite_config=ml_config_path,
    output_key="EventClassifierOutput",
)

tray.Add(
    "I3Writer",
    "writer",
    filename=str(outfile)+'.i3.zst',
    Streams=[
        icetray.I3Frame.TrayInfo,
        icetray.I3Frame.DAQ,
        icetray.I3Frame.Physics,
    ],
    DropOrphanStreams=[icetray.I3Frame.DAQ],
)

tray.Add("TrashCan", "trash")

try:
    tray.Execute()
    tray.Finish()
    print("Done:", infile.name)
except Exception as e:
    print("Failed:", infile.name)
    print(e)
    raise

print("Finished:", time.asctime())