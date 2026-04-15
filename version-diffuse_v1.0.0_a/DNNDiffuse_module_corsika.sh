#!/bin/bash

#prep inputs
infile=$1
outfile=$2
#setup env
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
source /data/user/zrechav/.venvs/alma_icetray/bin/activate

export HDF5_USE_FILE_LOCKING='FALSE'
echo ${infile}
echo ${outfile}
/data/user/zrechav/./i3/icetray/build/env-shell.sh python /data/user/zrechav/DNNDiffuse_Event_Selection/version-diffuse_v1.0.0_a/DNNDiffuse_module_corsika.py -i ${infile} -o ${outfile}  
