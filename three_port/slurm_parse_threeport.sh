
sbatch -o 3port_parse_${1}.out -e 3port_parse_${1}.err ./slurm_parse_threeport.sbatch ${1}
