#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH= 
conda deactivate
conda activate cotta 
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/source.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/norm.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/tent.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/cotta.yaml


