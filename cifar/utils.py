import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn as nn
import torch.utils.data as data
import logging
from tqdm import tqdm

def pytorch_evaluate(net: nn.Module, data_loader: data.DataLoader, fetch_keys: List,
                     x_shape: Tuple = None, output_shapes: Dict = None, to_tensor: bool=False, verbose=False) -> Tuple:

    if output_shapes is not None:
        for key in fetch_keys:
            assert key in output_shapes

    # Fetching inference outputs as numpy arrays
    batch_size = data_loader.batch_size
    num_samples = len(data_loader.dataset)
    batch_count = int(np.ceil(num_samples / batch_size))
    fetches_dict = {}
    fetches = []
    for key in fetch_keys:
        fetches_dict[key] = []

    net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (inputs, targets) in (tqdm(enumerate(data_loader)) if verbose else enumerate(data_loader)):
        if x_shape is not None:
            inputs = inputs.reshape(x_shape)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs_dict = net(inputs)
        for key in fetch_keys:
            fetches_dict[key].append(outputs_dict[key].data.cpu().detach().numpy())

    # stack variables together
    for key in fetch_keys:
        fetch = np.vstack(fetches_dict[key])
        if output_shapes is not None:
            fetch = fetch.reshape(output_shapes[key])
        if to_tensor:
            fetch = torch.as_tensor(fetch, device=torch.device(device))
        fetches.append(fetch)

    assert batch_idx + 1 == batch_count
    assert fetches[0].shape[0] == num_samples

    return tuple(fetches)

def boolean_string(s):
    # to use --use_bn True or --use_bn False in the shell. See:
    # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def convert_tensor_to_image(x: np.ndarray):
    """
    :param X: np.array of size (Batch, feature_dims, H, W) or (feature_dims, H, W)
    :return: X with (Batch, H, W, feature_dims) or (H, W, feature_dims) between 0:255, uint8
    """
    X = x.copy()
    X *= 255.0
    X = np.round(X)
    X = X.astype(np.uint8)
    if len(x.shape) == 3:
        X = np.transpose(X, [1, 2, 0])
    else:
        X = np.transpose(X, [0, 2, 3, 1])
    return X

def convert_image_to_tensor(x: np.ndarray):
    """
    :param X: np.array of size (Batch, H, W, feature_dims) between 0:255, uint8
    :return: X with (Batch, feature_dims, H, W) float between [0:1]
    """
    assert x.dtype == np.uint8
    X = x.copy()
    X = X.astype(np.float32)
    X /= 255.0
    X = np.transpose(X, [0, 3, 1, 2])
    return X

def majority_vote(x):
    return np.bincount(x).argmax()

def get_ensemble_paths(ensemble_dir):
    ensemble_subdirs = next(os.walk(ensemble_dir))[1]
    ensemble_subdirs.sort()
    ensemble_paths = []
    for j, dir in enumerate(ensemble_subdirs):  # for network j
        ensemble_paths.append(os.path.join(ensemble_dir, dir, 'ckpt.pth'))

    return ensemble_paths

def set_logger(log_file):
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)]
                        )

def print_Linf_dists(X, X_test):
    logger = logging.getLogger()
    X_diff = (X - X_test).reshape(X.shape[0], -1)
    X_diff_abs = np.abs(X_diff)
    Linf_dist = X_diff_abs.max(axis=1)
    Linf_dist = Linf_dist[np.where(Linf_dist > 0.0)[0]]
    logger.info('The adversarial attacks distance: Max[L_inf]={}, E[L_inf]={}'.format(np.max(Linf_dist), np.mean(Linf_dist)))

def calc_attack_rate(y_preds: np.ndarray, y_orig_norm_preds: np.ndarray, y_gt: np.ndarray) -> float:
    """
    Args:
        y_preds: The adv image's final prediction after the defense method
        y_orig_norm_preds: The original image's predictions
        y_gt: The GT labels
        targeted: Whether or not the attack was targeted
    
    Returns: attack rate in %
    """
    f0_inds = []  # net_fail
    f1_inds = []  # net_succ
    f2_inds = []  # net_succ AND attack_flip

    for i in range(len(y_gt)):
        f1 = y_orig_norm_preds[i] == y_gt[i]
        f2 = f1 and y_preds[i] != y_orig_norm_preds[i]
        if f1:
            f1_inds.append(i)
        else:
            f0_inds.append(i)
        if f2:
            f2_inds.append(i)

    attack_rate = len(f2_inds) / len(f1_inds)
    return attack_rate

def get_all_files_recursive(path, suffix=None):
    files = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(path):
        for file in f:
            if suffix is None:
                files.append(os.path.join(r, file))
            elif '.' + suffix in file:
                files.append(os.path.join(r, file))
    return files

def convert_grayscale_to_rgb(x: np.ndarray) -> np.ndarray:
    """
    Converts a 2D image shape=(x, y) to a RGB image (x, y, 3).
    Args:
        x: gray image
    Returns: rgb image
    """
    return np.stack((x, ) * 3, axis=-1)

def inverse_map(x: dict) -> dict:
    """
    :param x: dictionary
    :return: inverse mapping, showing for each val its key
    """
    inv_map = {}
    for k, v in x.items():
        inv_map[v] = k
    return inv_map

def get_image_shape(dataset: str) -> Tuple[int, int, int]:
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        return 32, 32, 3
    elif dataset == 'tiny_imagenet':
        return 64, 64, 3
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
