#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH= 
conda deactivate
conda activate cotta 
CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/source.yaml
CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/norm.yaml
CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/tent.yaml
CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/cotta.yaml


