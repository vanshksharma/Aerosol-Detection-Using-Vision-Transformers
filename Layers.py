import keras.layers
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow_probability as tfp
from utils import to_2tuple

np_config.enable_numpy_behavior()


def droppath(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob

    bernoulli_dist = tfp.distributions.Bernoulli(probs=keep_prob)
    shape = (x.shape[0],) + (1,) * (tf.rank(x).numpy() - 1)
    random_tensor = bernoulli_dist.sample(sample_shape=shape)

    if keep_prob > 0.0 and scale_by_keep:
        tf.divide(random_tensor, keep_prob)

    return x * random_tensor


class DropPath(keras.layers.Layer):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, **kwargs):
        return droppath(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = keras.layers.Dense(hidden_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = keras.layers.Dropout(drop_probs[0])
        self.fc2 = keras.layers.Dense(out_features, use_bias=bias[1])
        self.drop2 = keras.layers.Dropout(drop_probs[1])

    def call(self, x, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class Identity(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, **kwargs):
        return x


class Attention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(dim)
        self.proj_drop = keras.layers.Dropout(proj_drop)

    def call(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose((2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv, axis=0)

        attn = (q @ tf.experimental.numpy.swapaxes(k, -2, -1)) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.experimental.numpy.swapaxes((attn @ v), 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(keras.layers.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = tf.Variable(initial_value=init_values * tf.ones(dim), trainable=True)

    def call(self, x, **kwargs):
        if self.inplace:
            x = x * self.gamma
            return x

        return x * self.gamma


class Block(keras.layers.Layer):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=tf.nn.gelu, norm_layer=keras.layers.LayerNormalization):
        super().__init__()
        self.norm1 = norm_layer()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = norm_layer()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else Identity()

    def call(self, x, **kwargs):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
