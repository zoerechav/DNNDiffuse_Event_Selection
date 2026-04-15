import os
import sys
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="skip files that already exist")
args = parser.parse_args()

datasets = ['22803','23122','23123','23463']
#subdirs = ['0000000-0000999']
subdirs = [
    f"{i:07d}-{i+999:07d}"
    for i in range(0, 80000, 1000)
]


def get_Grid_job(infile,dataset,filenum,outfile,gcd, nugen,muongun,corsika,burn):
    if nugen==1:
        job_name = 'DNNDiffuse_' + dataset + '_' + filenum
        lines =[
            'JOB ' + job_name + ' /data/user/zrechav/DNNDiffuse_Event_Selection/version-diffuse_v1.0.0_a/DNNDiffuse_module_corsika.sub',
            'VARS ' + job_name + ' infile="' + infile + '" outfile="' + outfile +'" filenum="'+str(filenum)+'"',
            'Retry ' + job_name + ' 2',
            ]
        return lines


    
def get_Grid_jobs(input_path=None,output_path=None, gcd=None,nugen=1,muongun=0,corsika=0,burn=0,propagate=0):
    raw_infiles=sorted(glob.glob(input_path+ '/*.i3.zst'))
    #raw_infiles=raw_infiles[:10]
    #infile_numbers = [infile.split('/')[-1].split('_')[-1].split('.')[0] for infile in raw_infiles]
    infile_numbers = [infile.split('/')[-1].split('.')[0].split('_')[-1] for infile in raw_infiles]
    burn_sample_nums = []
    burn_sample = []
    for (num,file) in zip(infile_numbers,raw_infiles):
        #if int(num) % 10 == 0:
        burn_sample_nums.append(num)
        burn_sample.append(file)
    infile_numbers = burn_sample_nums
    raw_infiles = burn_sample
    #print(infile_numbers[0])
    #print(infile_numbers[-1])
    lines = []
    for (infile, filenum) in zip(raw_infiles,infile_numbers):
        
 
        outfile = output_path + dataset + '_' + filenum
    
        if args.resume:
            expected_output = outfile + ".i3.zst"
            if os.path.exists(expected_output):
                #print(f"Skipping existing file : {expected_output}")
                continue
        

        lines.extend(get_Grid_job(infile,dataset,filenum,outfile,gcd,nugen,muongun,corsika,burn))

    return lines


nugen=1

import os

dag_lines = []

if nugen == 1:
    gcd = os.path.join(
        "/cvmfs/icecube.opensciencegrid.org/data/GCDGeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz",
    )

    for dataset in datasets:
        for subdir in subdirs:

            input_path = os.path.join(
                "/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0",
                "DNNCascadeL4_monopod/Corsika",
                dataset,
                subdir,
            )

            # ---- skip if input doesn't exist ----
            if not os.path.isdir(input_path):
                continue
                
            print(f"input directory exists: {input_path}")
            output_path = os.path.join(
                "/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0",
                "DNNCascadeL5/version-diffuse_v1_0_0/Corsika",
                dataset,
                subdir,
                "DNNCascades_Diffuse_L5_",
            )
            
            output_dir = os.path.join(
            "/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0",
            "DNNCascadeL5/version-diffuse_v1_0_0/Corsika",
            dataset,
            subdir,
            )
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
                print(f"created {output_dir}")
            else:
                print(f"{output_dir} already exists")

            
            dag_lines.extend(
                get_Grid_jobs(
                    input_path=input_path,
                    output_path=output_path,
                    gcd=gcd,
                    nugen=1,
                    muongun=0,
                    corsika=0,
                    propagate=0,
                    burn=0,
                )
            )

#outfile_name='/home/zrechav/DAD/dags/DNNCascades_Diffuse_L5_' + dataset + '_.dag'
#outfile_name='/home/zrechav/DAD/dags/L4_preferred_fit_' + dataset + '_'+ subdir + '.dag'
outfile_name=f'/home/zrechav/npx/DNNDiffuse_Module/DNNDiffuse_L5_corsika_module.dag'
#outfile_name='/home/zrechav/DAD/dags/L5_og_DNNCascades_' + dataset + '_'+ subdir + '.dag'
with open(outfile_name, 'w') as f:
    f.write('\n'.join(dag_lines))

