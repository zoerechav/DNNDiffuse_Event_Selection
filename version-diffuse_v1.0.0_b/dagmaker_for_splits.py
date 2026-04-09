#!/usr/bin/env python

import os
import glob
import argparse

#datasets = ['22634', '22614','22633','22645','22646','22613']
datasets = ['22633','22634']

subdirs = [
    '0000000-0000999',
    '0001000-0001999',
]


def get_Grid_job(infile, dataset, filenum, outfile, gcd, nugen, muongun, corsika, burn):

    if nugen == 1:

        job_name = 'PF_MODULE_' + dataset + '_' + filenum

        lines = [
            'JOB ' + job_name + ' /data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/build_PF/npx_v2/v2/PF_MODULE.sub',
            'VARS ' + job_name + ' infile="' + infile + '" outfile="' + outfile + '" filenum="' + str(filenum) + '"',
            'Retry ' + job_name + ' 2',
        ]

        return lines

    return []


def extract_prefix_and_filenum(infile):
    """
    Example input filename:
    DNNCascades_Diffuse_L5_22613_00000000_000000.i3.zst

    filenum -> last two underscore-separated fields
    prefix  -> unsplit file index
    """

    base = os.path.basename(infile)
    stem = base.replace('.i3.zst', '')

    parts = stem.split('_')

    filenum = "_".join(parts[-2:])
    prefix = parts[-2]

    return prefix, filenum


def get_Grid_jobs(dataset, input_path=None, output_path=None, gcd=None,
                  nugen=1, muongun=0, corsika=0, burn=0, propagate=0,
                  max_prefix=None):

    raw_infiles = sorted(glob.glob(os.path.join(input_path, '*.i3.zst')))

    file_info = []
    for infile in raw_infiles:
        prefix, filenum = extract_prefix_and_filenum(infile)
        file_info.append((prefix, filenum, infile))

    # limit to first N unsplit prefixes
    if max_prefix is not None:

        unique_prefixes = sorted({prefix for prefix, _, _ in file_info})
        selected_prefixes = set(unique_prefixes[:max_prefix])

        file_info = [
            (prefix, filenum, infile)
            for prefix, filenum, infile in file_info
            if prefix in selected_prefixes
        ]

        print(f"Keeping first {len(selected_prefixes)} prefixes from {input_path}")
        print(f"Total split files selected: {len(file_info)}")

    lines = []

    for prefix, filenum, infile in file_info:

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--max_prefix',
        type=int,
        default=None,
        help='Maximum number of unsplit file prefixes (e.g. 100)'
    )

    parser.add_argument(
        '--outfile',
        default='/home/zrechav/npx/PF_MODULE/DNNDiffuse_L5_PF_split_more.dag',
        help='Output DAG filename'
    )

    args = parser.parse_args()

    max_prefix = args.max_prefix
    outfile_name = args.outfile

    nugen = 1
    muongun = 0
    corsika = 0
    burn = 0

    dag_lines = []

    if nugen == 1:

        gcd = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'

        for dataset in datasets:

            for subdir in subdirs:

                input_path = (
                    f'/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0/'
                    f'DNNCascadeL5/version-diffuse_v1_0_0_calc_PF/'
                    f'split_NuGen/{dataset}/{subdir}'
                )

                if not os.path.isdir(input_path):
                    print(f"Skipping missing input: {input_path}")
                    continue

                output_dir = (
                    f'/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0/'
                    f'DNNCascadeL5/version-diffuse_v1_0_0_calc_PF/'
                    f'to_stitch_NuGen/{dataset}/{subdir}'
                )

                os.makedirs(output_dir, exist_ok=True)

                output_path = output_dir + '/DNNCascades_Diffuse_PFModule_L5_'

                dag_lines.extend(
                    get_Grid_jobs(
                        dataset,
                        input_path=input_path,
                        output_path=output_path,
                        gcd=gcd,
                        nugen=1,
                        muongun=0,
                        corsika=0,
                        propagate=0,
                        burn=0,
                        max_prefix=max_prefix
                    )
                )

    with open(outfile_name, 'w') as f:
        f.write('\n'.join(dag_lines))

    print(f"Wrote {len(dag_lines)} DAG lines to {outfile_name}")


if __name__ == '__main__':
    main()