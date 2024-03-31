# CoTTA: Continual Test-Time Adaptation
Official code for [Continual Test-Time Domain Adaptation](https://arxiv.org/abs/2203.13591), published in CVPR 2022.

This repository also includes other continual test-time adaptation methods for classification and segmentation.
We provide benchmarking and comparison for the following methods:
+ [CoTTA](https://arxiv.org/abs/2203.13591) 
+ AdaBN / BN Adapt
+ TENT
  
on the following tasks
+ CIFAR10/100 -> CIFAR10C/100C (standard/gradual)
+ ImageNet -> ImageNetC
+ Cityscapes -> ACDC (segmentation)

## Prerequisite
Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate cotta 
```

## Classification Experiments
### CIFAR10-to-CIFAR10C-standard task
```bash
# Tested on RTX2080TI
cd cifar
# This includes the comparison of all three methods as well as baseline
bash run_cifar10.sh 
```
### CIFAR10-to-CIFAR10C-gradual task
```bash
# Tested on RTX2080TI
bash run_cifar10_gradual.sh
```
### CIFAR100-to-CIFAR100C task
```bash
# Tested on RTX3090
bash run_cifar100.sh
```

### ImageNet-to-ImageNetC task 
```bash
# Tested on RTX3090
cd imagenet
bash run.sh
```

## Segmentation Experiments
### Cityscapes-to-ACDC segmentation task
Since April 2022, we also offer the segmentation code based on Segformer.
You can download it [here](https://github.com/qinenergy/cotta/issues/6)
```
## environment setup: a new conda environment is needed for segformer
## You may also want to check https://github.com/qinenergy/cotta/issues/13 if you have problem installing mmcv
conda env create -f environment_segformer.yml
pip install -e . --user
conda activate segformer
## Run
bash run_base.sh
bash run_tent.sh
bash run_cotta.sh
# Example logs are included in ./example_logs/base.log, tent.log, and cotta.log.
## License for Cityscapses-to-ACDC code
Non-commercial. Code is heavily based on Segformer. Please also check Segformer's LICENSE.
```

## Data links
+ [Supplementary PDF](https://1drv.ms/b/s!At2KHTLZCWRegpAOqP-8BCBQze68wg?e=wiyaAl)
+ [ACDC experiment code](https://1drv.ms/u/s!At2KHTLZCWRegpAcvrh9SA34gMpzNQ?e=TSiKs6)
+ [Other Supplementary](https://1drv.ms/f/s!At2KHTLZCWRegpAKdcRuOGE1S9ZGLg?e=iJcR9L)
  
## Citation
Please cite our work if you find it useful.
```bibtex
@inproceedings{wang2022continual,
  title={Continual Test-Time Domain Adaptation},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgement 
+ TENT code is heavily used. [official](https://github.com/DequanWang/tent) 
+ KATANA code is used for augmentation. [official](https://github.com/giladcohen/KATANA) 
+ Robustbench [official](https://github.com/RobustBench/robustbench) 


## External data link
+ ImageNet-C [Download](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

For questions regarding the code, please contact wang@qin.ee .
