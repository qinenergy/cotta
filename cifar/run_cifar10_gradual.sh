#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=
conda deactivate
conda activate cotta

for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=0 python -u cifar10c_gradual.py --cfg cfgs/10orders/tent/tent$i.yaml
    CUDA_VISIBLE_DEVICES=0 python -u cifar10c_gradual.py --cfg cfgs/10orders/cotta/cotta$i.yaml
done



