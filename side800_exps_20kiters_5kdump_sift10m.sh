#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=03:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1
#SBATCH --exclude=nvl-[002,004] # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --tasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=40           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=100G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name sift80020k      # you can give your job a name for easier identification (same as -J)
#SBATCH --gres=gpu:v100:1
#SBATCH --gres-flags=enforce-binding

########## Command Lines to Run ##########

module load GCC/8.3.0 GCCcore/8.3.0 CUDA/10.1.243 OpenBLAS CMake Python/3.7.4 numactl ### load necessary modules, e.g.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/dikbayir/bin_gperf/lib
cd /mnt/home/dikbayir/stable-tsne/tsne-cuda                  ### change to the directory where your code is located

export OMP_NUM_THREADS=40

cd ./build/python/
pip3 install --user -e .

cd ../../
mkdir results_data
cd results_data


mkdir sift10m
cd sift10m
mkdir rabbit
cd rabbit
mkdir 800bin
cd 800bin
LD_PRELOAD=/mnt/home/dikbayir/bin_gperf/lib/libtcmalloc_minimal.so python ../../../../gen_syn_pts.py 10000000 50 15 150 6 2 4 20000 5000 20 800 2 1 5 > sift10m_20kiter_side800log.out

scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powetools command)
