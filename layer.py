import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from itertools import repeat
import collections.abc


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
