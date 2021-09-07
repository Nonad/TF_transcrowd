from functools import partial

import keras.layers
import numpy as np
from .register import register_model
from .layer import Mlp, DropPath, PatchEmbed
import tensorflow as tf
from tensorflow.keras import layers


DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
# timm/data/constants.py

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# 'default_cfgs'     is used in     register

class Attention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, activation=None, input_shape=(dim,), use_bias=True, bias_initializer=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, activation=None, input_shape=(dim,))
        self.proj_drop = layers.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = np.array(qkv)
        qkv = np.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.convert_to_tensor(qkv.transpose((2, 0, 3, 1, 4)))
        q, k, v = qkv[0], qkv[1], qkv[2]
        ksize = np.array(k.shape).size
        kperm = np.arange(0, ksize-1, 1)
        kperm = np.insert(kperm, -1, ksize-1)

        attn = (q @ tf.transpose(k, perm=kperm)) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        xsize = np.array(x.shape).size
        xperm = np.arange(2, xsize, 1)
        xperm = np.insert(xperm, 1, 1)
        xperm = np.insert(xperm, 0, 0)
        x = tf.convert_to_tensor(np.reshape(np.array(tf.transpose(x, perm=xperm)), (B, N, C)))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(keras.layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=layers.LayerNormalization()):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(layers.LayerNormalization, eps=1e-6)
        act_layer = act_layer or tf.nn.gelu

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token =