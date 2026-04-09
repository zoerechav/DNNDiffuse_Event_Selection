import os
import sys
import glob
import numpy as np

# datasets to process
datasets = ['22612']#, '22644', '22635']


# --------------------------------------------------
# Build subdirectory list like:
# 0000000-0000999 → 0024000-0024999
# --------------------------------------------------
def make_subdirs(start=0, stop=25000, step=1000):
    subdirs = []
    for i in range(start, stop, step):
        low = f"{i:07d}"
        high = f"{i + step - 1:07d}"
        subdirs.append(f"{low}-{high}")
    return subdirs


# --------------------------------------------------
# Single job definition
# --------------------------------------------------
def get_Grid_job(infile, dataset, filenum, outfile, gcd,
                 nugen, muongun, corsika, burn):

    if nugen == 1:
        job_name = 'PF_MODULE_' + dataset + '_' + filenum
        lines = [
            'JOB ' + job_name + ' /data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/build_PF/npx_v2/v2/PF_MODULE.sub',
            'VARS ' + job_name + f' infile="{infile}" outfile="{outfile}" filenum="{filenum}"',
            'Retry ' + job_name + ' 2',
        ]
        return lines

    return []


# --------------------------------------------------
# Build jobs for a directory
# --------------------------------------------------
def get_Grid_jobs(dataset,
                  input_path=None,
                  output_path=None,
                  gcd=None,
                  nugen=1,
                  muongun=0,
                  corsika=0,
                  burn=0,
                  propagate=0):

    raw_infiles = sorted(glob.glob(input_path + '/*.i3.zst'))

    if not raw_infiles:
        print(f"No files found in {input_path}")
        return []

    infile_numbers = [
        infile.split('/')[-1].split('.')[0].split('_')[-1]
        for infile in raw_infiles
    ]

    # (keeping your burn logic as-is)
    burn_sample_nums = []
    burn_sample = []
    for (num, file) in zip(infile_numbers, raw_infiles):
        burn_sample_nums.append(num)
        burn_sample.append(file)

    infile_numbers = burn_sample_nums
    raw_infiles = burn_sample

    lines = []

    for (infile, filenum) in zip(raw_infiles, infile_numbers):

        outfile = output_path + dataset + '_' + filenum

        lines.extend(
            get_Grid_job(
                infile,
                dataset,
                filenum,
                outfile,
                gcd,
                nugen,
                muongun,
                corsika,
                burn
            )
        )

    return lines


# --------------------------------------------------
# Main execution
# --------------------------------------------------
nugen = 1
muongun = 0
corsika = 0
burn = 0

dag_lines = []

if nugen == 1:

    gcd = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'

    subdirs = make_subdirs(2000, 25000, 1000)

    for dataset in datasets:
        for subdir in subdirs:

            input_path = (
                f'/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0/'
                f'DNNCascadeL5/version-diffuse_v1_0_0/NuGen/{dataset}/{subdir}'
            )

            # --- skip missing input directories ---
            if not os.path.isdir(input_path):
                print(f"Skipping missing input: {input_path}")
                continue

            # --- create output directory if needed ---
            output_dir = (
                f'/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0/'
                f'DNNCascadeL5/version-diffuse_v1_0_0_b/NuGen/{dataset}/{subdir}'
            )
            os.makedirs(output_dir, exist_ok=True)

            output_path = f'{output_dir}/DNNCascades_Diffuse_PFModule_L5_'

            dag_lines.extend(
                get_Grid_jobs(
                    dataset=dataset,
                    input_path=input_path,
                    output_path=output_path,
                    gcd=gcd,
                    nugen=nugen,
                    muongun=muongun,
                    corsika=corsika,
                    propagate=0,
                    burn=burn
                )
            )


# --------------------------------------------------
# Write DAG file
# --------------------------------------------------
outfile_name = f'/home/zrechav/npx/PF_MODULE/DNNDiffuse_L5_22612_PF.dag'

with open(outfile_name, 'w') as f:
    f.write('\n'.join(dag_lines))

print(f"Wrote DAG file: {outfile_name}")