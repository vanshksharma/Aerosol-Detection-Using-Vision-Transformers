import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
from Layers import DropPath, Identity
from utils import to_2tuple

np_config.enable_numpy_behavior()


def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / tf.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)], dtype=float)
    sinusoid_table[:, 0::2] = tf.math.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = tf.math.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return tf.expand_dims(sinusoid_table, axis=0)


class TokenPerformer(keras.layers.Layer):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt
        self.dim = dim
        self.kqv = keras.layers.Dense(3 * self.emb)
        self.dp = keras.layers.Dropout(dp1)
        self.proj = keras.layers.Dense(self.emb)
        self.head_cnt = head_cnt
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.epsilon = 1e-8

        self.mlp = keras.models.Sequential([
            keras.layers.Dense(1 * self.emb, activation=tf.nn.gelu),
            keras.layers.Dense(self.emb),
            keras.layers.Dropout(dp2)
        ])

        self.m = int(self.emb * kernel_ratio)
        initializer = keras.initializers.Orthogonal()
        self.w = tf.Variable(initial_value=initializer(shape=(self.m, self.emb)) * tf.math.sqrt(self.m),
                             trainable=False)

    def prm_exp(self, x):
        xd = tf.experimental.numpy.tile(tf.reduce_sum(x * x, axis=-1, keepdims=True), (1, 1, self.m)) / 2
        wtx = tf.einsum("bti,mi->btm", tf.cast(x, dtype=tf.float32), self.w)

        return tf.exp(wtx - xd) / tf.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = tf.split(self.kqv(x), self.emb, axis=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = tf.expand_dims(tf.einsum("bti,bi->bt", qp, tf.reduce_sum(kp, axis=1)), axis=2)
        kptv = tf.einsum("bin,bim->bnm", tf.cast(v, dtype=tf.float32), kp)
        y = tf.einsum("bti,bni->btn", qp, kptv) / (tf.experimental.numpy.tile(D, (1, 1, self.emb)) + self.epsilon)
        y = v + self.dp(self.proj(y))

        return y

    def call(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Mlp(keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = keras.layers.Dense(hidden_features)
        self.act = act_layer
        self.fc2 = keras.layers.Dense(out_features)
        self.drop = keras.layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = keras.layers.Dense(in_dim * 3, use_bias=qkv_bias)
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(in_dim)
        self.proj_drop = keras.layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ tf.experimental.numpy.swapaxes(k, -2, -1)) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.experimental.numpy.swapaxes((attn @ v), 1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = tf.squeeze(v, axis=1) + x

        return x


class TokenTransformer(keras.layers.Layer):
    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=keras.layers.LayerNormalization):
        super().__init__()
        self.norm1 = norm_layer()
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer()
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class T2T(keras.layers.Layer):
    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if patch_size == 12:
            kernel_size = ((7, 4, 2), (3, 3, 1), (3, 1, 1))
        elif patch_size == 16:
            kernel_size = ((7, 4, 2), (3, 2, 1), (3, 2, 1))
        else:
            raise ValueError(f"Unknown patch size {patch_size}")

        self.soft_split0 = lambda x: tf.image.extract_patches(x, sizes=(1,) + to_2tuple(kernel_size[0][0]) + (1,),
                                                              strides=(1,) + to_2tuple(kernel_size[0][1]) + (1,),
                                                              rates=[1, 1, 1, 1],
                                                              padding=(1,) + to_2tuple(kernel_size[0][2]) + (1,))

        self.soft_split1 = lambda x: tf.image.extract_patches(x, sizes=(1,) + to_2tuple(kernel_size[1][0]) + (1,),
                                                              strides=(1,) + to_2tuple(kernel_size[1][1]) + (1,),
                                                              rates=[1, 1, 1, 1],
                                                              padding=(1,) + to_2tuple(kernel_size[1][2]) + (1,))

        self.soft_split2 = lambda x: tf.image.extract_patches(x, sizes=(1,) + to_2tuple(kernel_size[2][0]) + (1,),
                                                              strides=(1,) + to_2tuple(kernel_size[2][1]) + (1,),
                                                              rates=[1, 1, 1, 1],
                                                              padding=(1,) + to_2tuple(kernel_size[2][2]) + (1,))
        if tokens_type == 'transformer':
            self.attention1 = TokenTransformer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim, num_heads=1,
                                               mlp_ratio=1.0)
            self.attention2 = TokenTransformer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim, num_heads=1,
                                               mlp_ratio=1.0)
            self.project = keras.layers.Dense(embed_dim)

        elif tokens_type == 'performer':
            self.attention1 = TokenPerformer(dim=in_chans * (kernel_size[0][0] ** 2),
                                             in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = TokenPerformer(dim=token_dim * (kernel_size[1][0] ** 2),
                                             in_dim=token_dim, kernel_ratio=0.5)
            self.project = keras.layers.Dense(embed_dim)

        self.num_patches = (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1])) * \
                           (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1]))
