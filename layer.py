import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from itertools import repeat
import collections.abc
import math
import warnings

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, activation=None, input_shape=(in_features,))
        self.act = act_layer()
        self.fc2 = layers.Dense(out_features, activation=None, input_shape=(hidden_features,))
        self.drop = layers.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    dim = np.array(x).ndim
    shape = (x.shape[0], ) + (1, ) * (dim - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=tf.constant(x).dtype)
    random_tensor.__floor__()
    output = tf.divide(x, keep_prob) * random_tensor
    return output


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob


    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else tf.identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
          f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            xdim = tf.reduce_prod(tf.shape(x)[2:])        # [a,b,c*d]
            x = tf.reshape(x, [-1, xdim])     # -1 all
            xsize = np.array(x.shape).size
            xperm = np.arange(2, xsize, 1)
            xperm = np.insert(xperm, 1, 1)
            xperm = np.insert(xperm, 0, 0)
            x = tf.transpose(x, perm=xperm)
        x = self.norm(x)
        return x

# timm/models/layers/weight_init.py
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # (tensor, float, float, float, float) -> tensor
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)
#
# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     def norm_cdf(x):
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.
#
#     if(mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)
#
#     with tf.GradientTape(watch_accessed_variables=False) as tape:
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#
#         tensor = tf.random.uniform(tensor.shape, 2 * l - 1, 2 * u - 1, Dtype=tf.float32)
#
#         tensor = tf.math.erfinv(tensor)
#
#         tensor = tf.add(tf.multiply(tensor, std * math.sqrt(2.)), mean)
#
#         tensor = tf.clip_by_value(tensor, min=a, max=b)
#         return tensor
