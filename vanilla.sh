#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=01:40:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1
#SBATCH --nodelist=nvl-007           # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --tasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=32G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name reorderedCUDA      # you can give your job a name for easier identification (same as -J)
#SBATCH --gres=gpu:v100:1
#SBATCH --gres-flags=enforce-binding

########## Command Lines to Run ##########
 
module load GCC/8.3.0 GCCcore/8.3.0 CUDA/10.1.243 OpenBLAS CMake Python/3.7.4  ### load necessary modules, e.g.
 
cd /mnt/home/dikbayir/perm-tsne/tsnecuda/                   ### change to the directory where your code is located

python ./gen_syn_pts.py 2000000 50 15 150 1 0 4 > warmup406.out
python ./gen_syn_pts.py 8000000 50 15 150 1 0 4 > vanilla_out406.out ### call your executable (similar to mpirun)

python ./gen_syn_pts.py 8000000 50 15 150 1 1 8 > reord406.out

python ./gen_syn_pts.py 16000000 50 15 150 1 1 4 > reord16m406.out
python ./gen_syn_pts.py 16000000 50 15 150 1 0 4 > vanilla16m406.out

scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powetools command)
