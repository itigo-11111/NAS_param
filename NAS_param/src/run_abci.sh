#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

umask 007

source /etc/profile.d/modules.sh
source /home/aad13659kt/anaconda3/bin/activate henmi

WORKDIR='/groups/gaa50073/henmi-kazuki/NAS_param/'

# network='resnet50'
# epochs=50
# dataset='dataset_guliver_front_back'
dataset='cifar10'

cd $WORKDIR

mkdir $SGE_LOCALDIR/dataset/
cp ${dataset}.tar.gz $SGE_LOCALDIR/dataset/
cd $SGE_LOCALDIR/dataset/
tar -I pigz -xf ${dataset}.tar.gz


cd $WORKDIR


limit_param = 3000000
lambda_a = 0.01
gammas_learning_rate = 6e-1


export CUDA_VISIBLE_DEVICES="0"
python src/train_search.py --set=${dataset} --data=$SGE_LOCALDIR/dataset/${dataset} --limit_param=${limit_param} --lambda_a=${lambda_a} --gammas_learning_rate=${gammas_learning_rate} --id=1&
# python src/main.py --dataset=${dataset} --path2db=$SGE_LOCALDIR/dataset/${dataset}/

lambda_a = 0.001
export CUDA_VISIBLE_DEVICES="1"
python src/train_search.py --set=${dataset} --data=$SGE_LOCALDIR/dataset/${dataset} --limit_param=${limit_param} --lambda_a=${lambda_a} --gammas_learning_rate=${gammas_learning_rate} --id=2&
# python src/main.py --dataset=${dataset} --path2db=$SGE_LOCALDIR/dataset/${dataset}/

lambda_a = 0.0001
export CUDA_VISIBLE_DEVICES="2"
python src/train_search.py --set=${dataset} --data=$SGE_LOCALDIR/dataset/${dataset} --limit_param=${limit_param} --lambda_a=${lambda_a} --gammas_learning_rate=${gammas_learning_rate} --id=3&
# python src/main.py --dataset=${dataset} --path2db=$SGE_LOCALDIR/dataset/${dataset}/

lambda_a = 0.00001
export CUDA_VISIBLE_DEVICES="3"
python src/train_search.py --set=${dataset} --data=$SGE_LOCALDIR/dataset/${dataset} --limit_param=${limit_param} --lambda_a=${lambda_a} --gammas_learning_rate=${gammas_learning_rate} --id=4&
# python src/main.py --dataset=${dataset} --path2db=$SGE_LOCALDIR/dataset/${dataset}/


wait
