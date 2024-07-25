#!/bin/bash
#SBATCH --nodes=1                   #####Nb de noeuds
#SBATCH --ntasks-per-node=6        ##### Nb de coeurs/noeuds ici 24
#SBATCH --time=24:00:00             ##### temps demandé maximum de mon calcul
#SBATCH --mail-user=avani.malaviya@isae-supaero.fr #### mon mail
#SBATCH --mail-type=FAIL,END                   #### mail quand c'est fini ou erreur
#SBATCH --job-name=mmdet_trial                   #### mon calcul
#SBATCH -o    slurm.%j.out                     #### sorti standard STDOUT
#SBATCH -e    slurm.%j.err                     #### sorti erreur STDERR
#SBATCH --partition=gpu                        #### la partition
#SBATCH --gres=gpu:1

####  loading python you want 2.7 or 3.7 ###
module purge
module load python/2023-3
module list

##### Activate your environement ######
conda deactivate 
source activate mmlab

export https_proxy=http://proxy.isae.fr:3128


#### #launch your program #####
cd /scratch/disc/a.malaviya/mmdetection
python tools/misc/download_dataset.py --dataset-name balloon --save-dir data --unzip
python mmdet_prepare_balloon_dataset.py
echo !!!!!!!!!!!!!!!!!!!!!!!! training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python tools/train.py configs/rtmdet/rtmdet_tiny_1xb4-20e_balloon.py
echo !!!!!!!!!!!!!!!!!!!!!!!! infering !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python mmdet_infer_balloon.py

