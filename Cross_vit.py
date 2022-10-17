import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config

from Layers import DropPath, Mlp, Identity, Block
from utils import to_2tuple

np_config.enable_numpy_behavior()


# INITIALIZER=keras.initializers.TruncatedNormal(stddev=0.02)


class PatchEmbed(keras.layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        self.in_chans = in_chans
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = keras.models.Sequential([
                    keras.layers.ZeroPadding2D(padding=3),
                    keras.layers.Conv2D(embed_dim // 4, kernel_size=7, strides=4, activation="relu"),
                    keras.layers.Conv2D(embed_dim // 2, kernel_size=3, strides=3, activation="relu"),
                    keras.layers.ZeroPadding2D(padding=1),
                    keras.layers.Conv2D(embed_dim, kernel_size=3, strides=1, activation=None),
                ])

            elif patch_size[0] == 16:
                self.proj = keras.models.Sequential([
                    keras.layers.ZeroPadding2D(padding=3),
                    keras.layers.Conv2D(embed_dim // 4, kernel_size=7, strides=4, activation="relu"),
                    keras.layers.ZeroPadding2D(padding=1),
                    keras.layers.Conv2D(embed_dim // 2, kernel_size=3, strides=2, activation="relu"),
                    keras.layers.ZeroPadding2D(padding=1),
                    keras.layers.Conv2D(embed_dim, kernel_size=3, strides=2, activation=None),
                ])

        else:
            self.proj = keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, activation=None)

    def call(self, x):

        B, H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(x, (tf.shape(x).numpy()[0], -1, tf.shape(x).numpy()[3]))

        return x


class CrossAttention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = keras.layers.Dense(dim, use_bias=qkv_bias)
        self.wk = keras.layers.Dense(dim, use_bias=qkv_bias)
        self.wv = keras.layers.Dense(dim, use_bias=qkv_bias)
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(dim)
        self.proj_drop = keras.layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).transpose((0, 2, 1, 3))
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose((0, 2, 1, 3))
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose((0, 2, 1, 3))

        attn = (q @ k.transpose((-2, -1))) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose((1, 2)).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=keras.layers.LayerNormalization, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer()
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer()
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def call(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(keras.layers.Layer):
    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=keras.layers.LayerNormalization):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        self.blocks = []
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))

            if len(tmp) != 0:
                self.blocks.append(keras.models.Sequential(tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = []
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [Identity()]
            else:
                tmp = [norm_layer(), act_layer(), keras.layers.Dense(dim[(d + 1) % num_branches])]
            self.projs.append(keras.models.Sequential(tmp))

        self.fusion = []
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:
                self.fusion.append(
                    CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                        has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                   norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(keras.models.Sequential(tmp))

        self.revert_projs = []
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [Identity()]
            else:
                tmp = [norm_layer(), act_layer(), keras.layers.Dense(dim[d])]
            self.revert_projs.append(keras.models.Sequential(tmp))

    def call(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        outs = []
        for i in range(self.num_branches):
            tmp = tf.concat([proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]], axis=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = tf.concat([reverted_proj_cls_token, outs_b[i][:, 1:, ...]], axis=1)
            outs.append(tmp)

        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size, patches)]


class VisionTransformer(keras.models.Model):
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=keras.layers.LayerNormalization, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)
        self.patch_embed = []
        # if hybrid_backbone is None:
        self.pos_embed = [
            tf.Variable(initial_value=tf.zeros(shape=(1, 1 + num_patches[i], embed_dim[i])), trainable=True)
            for i in range(self.num_branches)]
        for im_s, p, d in zip(img_size, patch_size, embed_dim):
            self.patch_embed.append(
                PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))

        self.cls_token = [tf.Variable(initial_value=tf.zeros(shape=(1, 1, embed_dim[i])), trainable=True) for i in
                          range(self.num_branches)]
        self.pos_drop = keras.layers.Dropout(drop_rate)
        total_depth = tf.reduce_sum([tf.reduce_sum(x[-2:]) for x in depth]).numpy()
        dpr = [x.numpy() for x in tf.linspace(tf.constant(0.), tf.constant(drop_path_rate), tf.constant(total_depth))]
        dpr_ptr = 0
        self.blocks = []
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = [norm_layer(input_shape=embed_dim[i]) for i in range(self.num_branches)]
        self.head = [keras.layers.Dense(num_classes, input_shape=embed_dim[i]) if num_classes > 0 else Identity() for i
                     in range(self.num_branches)]

        for i in range(self.num_branches):
            if self.pos_embed[i].trainable:
                self.pos_embed[i].assign(tf.random.truncated_normal(shape=self.pos_embed[i].shape, stddev=0.02))
            self.cls_token[i].assign(tf.random.truncated_normal(shape=self.cls_token[i].shape, stddev=0.02))

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = tf.image.resize(x, size=(self.img_size[i], self.img_size[i]),
                                 method=tf.image.ResizeMethod.BICUBIC) if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = tf.experimental.numpy.tile(self.cls_token[i], (B, 1, 1))
            tmp = tf.concat([cls_tokens, tmp], axis=1)
            tmp += self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def call(self, x):
        xs = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = tf.reduce_mean(tf.stack(ce_logits, axis=0), axis=0)

        return ce_logits
