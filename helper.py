import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable
import math

import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2
import random
import os

def setup_seed(SEED):
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED) # if you are using multi-GPU.
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

def load_data(img_path, args, train=True):

    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            gt_count = np.asarray(gt_file['gt_count'])
            break  # Success!
        except OSError:
            print("load error:", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count



def named_apply(fn: Callable, module: layers.Layer, name='', depth_first=True, include_root=False) -> layers.Layer:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# this function is copied from timm/models/helpers.py
def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight