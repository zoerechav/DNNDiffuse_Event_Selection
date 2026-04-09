import os
import sys
import glob
import numpy as np

#dataset = '22612'
datasets = ['22612','22644','22635','22613','22614','22633','22634','22645','22646']
subdir = '0000000-0000999'

def get_Grid_job(infile,dataset,filenum,outfile,gcd, nugen,muongun,corsika,burn):
    if nugen==1:
        job_name = 'PF_MODULE_' + dataset + '_' + filenum
        lines =[
            'JOB ' + job_name + ' /data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/build_PF/npx_v2/v2/PF_MODULE.sub',
            'VARS ' + job_name + ' infile="' + infile + '" outfile="' + outfile +'" filenum="'+str(filenum)+'"',
            'Retry ' + job_name + ' 2',
            ]
        return lines


    
def get_Grid_jobs(input_path=None,output_path=None, gcd=None,nugen=1,muongun=0,corsika=0,burn=0,propagate=0):
    raw_infiles=sorted(glob.glob(input_path+ '/*.i3.zst'))
    #raw_infiles=raw_infiles[:100]
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
        

        lines.extend(get_Grid_job(infile,dataset,filenum,outfile,gcd,nugen,muongun,corsika,burn))

    return lines


nugen=1
muongun=0

corsika=0
burn =0
nfiles=1000
dag_lines = []
if nugen ==1:
    gcd = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'
    
    for dataset in datasets:
        input_path = f'/data/ana/Diffuse/DNNCascades_Diffuse/version-1.0/DNNCascadeL5/version-diffuse_v1_0_0_b/NuGen/{dataset}/{subdir}'

        #/data/user/zrechav/DNNCascades_Diffuse/bdt_selfveto/bdt_scripts/build_PF/i3_pf/22613
        output_path = f'/data/user/zrechav/DNNCascades_Diffuse/gompert_test/NuGen/{dataset}/{subdir}/DNNCascades_Diffuse_PFModule_L5_'
        dag_lines.extend(
        get_Grid_jobs(
            input_path=input_path,
            output_path = output_path,
            gcd=gcd,
            nugen=1,
            muongun=0,corsika=0,propagate=0,burn=0
            )
        )
#outfile_name='/home/zrechav/DAD/dags/DNNCascades_Diffuse_L5_' + dataset + '_.dag'
#outfile_name='/home/zrechav/DAD/dags/L4_preferred_fit_' + dataset + '_'+ subdir + '.dag'
outfile_name=f'/home/zrechav/npx/PF_MODULE/DNNDiffuse_gompert_{subdir}.dag'
#outfile_name='/home/zrechav/DAD/dags/L5_og_DNNCascades_' + dataset + '_'+ subdir + '.dag'
with open(outfile_name, 'w') as f:
    f.write('\n'.join(dag_lines))

