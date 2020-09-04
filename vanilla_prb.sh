#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=01:50:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1
#SBATCH --exclude=nvl-[002,004] # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --tasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=40           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=32G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name reorderedCUDA      # you can give your job a name for easier identification (same as -J)
#SBATCH --gres=gpu:v100:1
#SBATCH --gres-flags=enforce-binding

########## Command Lines to Run ##########
 
module load GCC/8.3.0 GCCcore/8.3.0 CUDA/10.1.243 OpenBLAS CMake Python/3.7.4 numactl ### load necessary modules, e.g.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/dikbayir/bin_gperf/lib
cd /mnt/home/dikbayir/perm-tsne/tsne-cuda/                   ### change to the directory where your code is located

cd ./build/python/
pip3 install --user -e .

cd ../../
mkdir prb_3m
cd prb_3m
mkdir vanilla
cd vanilla

export OMP_NUM_THREADS=40

LD_PRELOAD=/mnt/home/dikbayir/bin_gperf/lib/libtcmalloc_minimal.so python ../../gen_syn_pts.py 3000000 50 15 150 2 0 4 5000 1000 10 > prb10_v100_3m_5k_vanilla.out

cd ..
mkdir rabbit
cd rabbit

LD_PRELOAD=/mnt/home/dikbayir/bin_gperf/lib/libtcmalloc_minimal.so python ../../gen_syn_pts.py 3000000 50 15 150 2 2 4 5000 1000 10 > prb10_v100_3m_5k_rab.out

cd ..
mkdir rabbit300
cd rabbit300

LD_PRELOAD=/mnt/home/dikbayir/bin_gperf/lib/libtcmalloc_minimal.so python ../../gen_syn_pts.py 3000000 300 15 150 2 2 4 5000 1000 10 > prb10_v100_3m300_5k_rab.out



scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powetools command)
