#!/bin/bash
# slurm_parse_threeport.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-12:00 # Running time of 3 hours
#SBATCH --mem=8132 #16384 #70656 # Memory request of 70 GB (inc to 98304)

# load modules
module load centos6/0.0.1-fasrc01
#module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/behavior

# grab filename from array exported from 'parent' shell:
#FILENAME="$1"
#echo "File: ${FILENAME}"

echo ${1}

# run processing on raw data
python ./parse_session_data.py -c ${1} -p threeport -n 1 --new --process #--plot-session 
