from .model import base_patch16_384_gap, base_patch16_384_token
from __future__ import division
import tensorflow as tf
from keras import layers
import nni
import logging
from nni.utils import merge_parameter
import os
from .config import return_args, args
import numpy as np
# from utils import save_checkpoint, setup_seed
# save pth...
# from image import load_data
from .helper import load_data, setup_seed
import time
import warnings

warnings.filterwarnings('ignore')

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'
