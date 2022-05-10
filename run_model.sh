#!/bin/bash
#SBATCH --partition=aa100-ucb
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --ntasks=64

module purge
module load cuda/11.2
source /curc/sw/anaconda3/latest
conda env create --file environment.yml #activate yolact3
conda activate yolact3

export WANDB_API_KEY=PASTE_KEY_KEY
fold_num=3
shot_num=10
resume_model=weights/___.pth

base_config=fold_${fold_num}_base_config_xxx
fine_tune_config=fold_${fold_num}_fine_tune_${shot_num}_shot_config_xxx

# train base model
python train.py --config=${base_config} --batch_size=24 --validation_size 3 --validation_epoch 1000 --save_interval 2000

# resume base model training
# python train.py --config=${base_config} --batch_size=24 --validation_size 3 --validation_epoch 1000 --save_interval 2000 --resume=${resume_model} --start_iter=-1

# fine-tune k-shot
# python train.py --config=${fine_tune_config} --batch_size=24 --validation_size 1 --validation_epoch 1000 --save_interval 2000 --resume=${resume_model} --start_iter=1