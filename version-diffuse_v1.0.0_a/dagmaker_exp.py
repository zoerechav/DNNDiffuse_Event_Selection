import os
import sys
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    action="store_true",
    help="skip files that already exist (checks for <outfile>.i3.zst)",
)
args = parser.parse_args()

#datasets = ['23521','23522','23523','23524','23525','23526','23527','23528','23529']
datasets = ['IC86_2017','IC86_2018']


def get_Grid_job(infile, dataset, filenum, outfile, gcd, nugen, muongun, corsika, burn):
    if nugen == 1:
        job_name = 'DNNDiffuse_' + dataset + '_' + filenum
        lines = [
            'JOB ' + job_name + ' /data/user/zrechav/DNNDiffuse_Event_Selection/version-diffuse_v1.0.0_a/DNNDiffuse_module_exp.sub',
            'VARS ' + job_name + ' infile="' + infile + '" outfile="' + outfile + '" filenum="' + str(filenum) + '"',
            'Retry ' + job_name + ' 2',
        ]
        return lines
    return []

def get_Grid_jobs(
    input_path=None,
    output_path=None,
    output_dir=None,
    gcd=None,
    nugen=1,
    muongun=0,
    corsika=0,
    burn=0,
    propagate=0,
):
    raw_infiles = sorted(glob.glob(os.path.join(input_path, "*.i3.zst")))
    infile_numbers = [infile.split('/')[-1].split('.')[0].split('_')[-1] for infile in raw_infiles]
    burn_sample_nums = []
    burn_sample = []
    for (num, file) in zip(infile_numbers, raw_infiles):
        if int(num) % 10 == 0:
            burn_sample_nums.append(num)
            burn_sample.append(file)
    infile_numbers = burn_sample_nums
    raw_infiles = burn_sample

    lines = []
    for (infile, filenum) in zip(raw_infiles, infile_numbers):

        outfile = output_path + dataset + '_' + filenum  # keep your existing naming convention

        # --- resume: skip if expected output exists ---
        if args.resume:
            expected_output = outfile + ".i3.zst"
            if os.path.exists(expected_output):
                continue

        # --- make output directory iff we actually have at least one input file to process ---
        # (and only when we aren't skipping due to resume)
        if output_dir is not None and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"created {output_dir}")

        lines.extend(get_Grid_job(infile, dataset, filenum, outfile, gcd, nugen, muongun, corsika, burn))

    return lines


nugen = 1
dag_lines = []

if nugen == 1:
    gcd = os.path.join(
        "/cvmfs/icecube.opensciencegrid.org/data/GCDGeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz",
    )

    for dataset in datasets:
        

        input_path = os.path.join(
            "/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0",
            "DNNCascadeL4_monopod/exp",
            dataset,
           
        )

        # ---- skip if input doesn't exist ----
        if not os.path.isdir(input_path):
            continue

        # ---- skip if there are no input files ----
        # (so we don't create empty output dirs)
        n_inputs = len(glob.glob(os.path.join(input_path, "*.i3.zst")))
        if n_inputs == 0:
            continue

        print(f"input directory exists: {input_path} ({n_inputs} files)")

        output_dir = os.path.join(
            "/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0",
            "DNNCascadeL5/version-diffuse_v1_0_0/exp",
            dataset,
        )

        output_path = os.path.join(
            output_dir,
            "DNNCascades_Diffuse_v1_0_0_a_",
        )

        dag_lines.extend(
            get_Grid_jobs(
                input_path=input_path,
                output_path=output_path,
                output_dir=output_dir,
                gcd=gcd,
                nugen=1,
                muongun=0,
                corsika=0,
                propagate=0,
                burn=0,
            )
        )

outfile_name = "/home/zrechav/npx/DNNDiffuse_Module/DNNDiffuse_L5_exp.dag"
with open(outfile_name, "w") as f:
    f.write("\n".join(dag_lines))