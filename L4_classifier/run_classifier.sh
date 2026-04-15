#!/bin/bash

#prep inputs
infile=$1
outfile=$2
#setup env
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh`

export HDF5_USE_FILE_LOCKING='FALSE'
echo ${infile}
echo ${outfile}
/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/RHEL_9_x86_64_v2/metaprojects/icetray/v1.17.0/bin/icetray-shell python /data/user/zrechav/DNNDiffuse_Event_Selection/L4_classifier/run_classifier.py --infile ${infile} --outfile ${outfile}  
