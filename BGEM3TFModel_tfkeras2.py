from typing import Dict, List, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

def gelu_exact(x):
    # exact erf-based GELU
    return tf.nn.gelu(x, approximate=False)

# --- 맨 위에 추가 ---
class TorchLayerNorm(layers.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-5, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(self.hidden_size,), initializer="ones", dtype=tf.float32, trainable=True)
        self.beta = self.add_weight(name="beta",  shape=(self.hidden_size,), initializer="zeros", dtype=tf.float32, trainable=True)
        super().build(input_shape)

    def call(self, x):
        # PyTorch LayerNorm과 동일: 모집단 분산(mean of squares) 사용
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        xhat = (x - mean) / tf.sqrt(var + self.eps)
        return xhat * self.gamma + self.beta


class TorchLayerNormTF1:
    def __init__(self, hidden_size: int, eps: float = 1e-5, scope: str = "LayerNorm"):
        self.hidden_size = hidden_size
        self.eps = eps
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            self.gamma = tf.compat.v1.get_variable(
                "weight",
                shape=[hidden_size],
                initializer=tf.compat.v1.initializers.ones(),
                dtype=tf.float32,
            )
            self.beta = tf.compat.v1.get_variable(
                "bias",
                shape=[hidden_size],
                initializer=tf.compat.v1.initializers.zeros(),
                dtype=tf.float32,
            )

    def __call__(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        xhat = (x - mean) / tf.sqrt(var + self.eps)
        return xhat * self.gamma + self.beta


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model, use_bias=True, name=f"{self.name}.wq")
        self.wk = layers.Dense(d_model, use_bias=True, name=f"{self.name}.wk")
        self.wv = layers.Dense(d_model, use_bias=True, name=f"{self.name}.wv")
        self.dense = layers.Dense(d_model, use_bias=True, name=f"{self.name}.dense")

        self.attlayerNorm = TorchLayerNormTF1(self.d_model, eps=1e-5, scope=f"{self.name}.attlayerNorm")
        self.dropout = layers.Dropout(rate=dropout_rate)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=False):
        bsz = tf.shape(inputs)[0]
        q = self.split_heads(self.wq(inputs), bsz)
        k = self.split_heads(self.wk(inputs), bsz)
        v = self.split_heads(self.wv(inputs), bsz)

        dk = tf.cast(self.depth, tf.float32)
        attn_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk)
        if mask is not None:
            attn_scores = attn_scores + mask

        # stable softmax (+1e-9) to mirror the working code
        attn_probs = tf.nn.softmax(attn_scores + 1e-9, axis=-1)
        attn_probs = self.dropout(attn_probs, training=training)

        ctx = tf.matmul(attn_probs, v)
        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        ctx = tf.reshape(ctx, (bsz, -1, self.d_model))

        out = self.dense(ctx)
        if training:
            out = self.dropout(out, training=training)
        out = self.attlayerNorm(out + inputs)
        return out

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, intermediate_size: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, name=f"{self.name}.attention")
        self.intermediate = layers.Dense(intermediate_size, use_bias=True, name=f"{self.name}.intermediate.dense")
        self.output_dense = layers.Dense(d_model, use_bias=True, name=f"{self.name}.output.dense")
        self.output_dropout = layers.Dropout(dropout_rate)
        self.output_norm = TorchLayerNormTF1(d_model, eps=1e-5, scope=f"{self.name}.output.LayerNorm")

    def call(self, x, attention_mask=None, training=False):
        x_att = self.attention(x, mask=attention_mask, training=training)
        inter = self.intermediate(x_att)
        inter = gelu_exact(inter)
        out = self.output_dense(inter)
        if training:
            out = self.output_dropout(out, training=training)
        x_out = self.output_norm(out + x_att)
        return x_out

class BGEM3TensorFlow(Model):
    def __init__(
        self,
        vocab_size: int = 250002,
        max_position_embeddings: int = 8194,
        type_vocab_size: int = 1,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        dropout_rate: float = 0.0,
        name: str = "bge-m3-tensorflow",
    ):
        super().__init__(name=name)
        self.padding_idx = 1
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers

        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="embeddings", shape=[vocab_size, hidden_size],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32
            )
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings", shape=[max_position_embeddings, hidden_size],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32
            )
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings", shape=[type_vocab_size, hidden_size],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32
            )

        # 이름은 HF와 다를 수 있어도 상관없음. 가중치만 정확히 들어가면 됩니다.
        self.layerNorm = TorchLayerNormTF1(hidden_size, eps=1e-5, scope="embeddings.LayerNorm")
        self.dropout = layers.Dropout(rate=dropout_rate)

        self.encoder_layers: List[TransformerBlock] = []
        for i in range(num_hidden_layers):
            self.encoder_layers.append(
                TransformerBlock(
                    d_model=hidden_size,
                    num_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    dropout_rate=dropout_rate,
                    name=f"encoder.layer.{i}",
                )
            )

        # pooler는 내보내기 시점 디버그용으로만 빌드. dense_vecs는 raw CLS 반환.
        self.pooler = layers.Dense(hidden_size, activation="tanh", use_bias=True, name="pooler.dense")
        self.colbert_linear = layers.Dense(units=hidden_size, use_bias=True, name="colbert_linear")

    # 경고 제거용
    def build(self, input_shape):
        self.built = True

    @staticmethod
    def _create_position_ids_from_attention_mask(attention_mask, padding_idx=1, past_key_values_length=0):
        # attention_mask: [B,T] 0/1
        mask = tf.cast(attention_mask, tf.int32)
        incremental = tf.math.cumsum(mask, axis=1)
        if past_key_values_length != 0:
            incremental = incremental + tf.cast(past_key_values_length, tf.int32)
        # pads -> padding_idx, tokens -> cumsum + padding_idx
        return incremental * mask + tf.cast(padding_idx, tf.int32)

    @staticmethod
    def _shape_list(t: Union[tf.Tensor, np.ndarray]) -> List[int]:
        if isinstance(t, np.ndarray):
            return list(t.shape)
        dynamic = tf.shape(t)
        static = t.shape.as_list()
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    # === HF 로직과 완전히 동일하게 다시 작성 (형/연산 순서 포함) ===
    @staticmethod
    def _create_position_ids_from_input_ids(input_ids, padding_idx=1, past_key_values_length=0):
        # Replace non-padding symbols with their position numbers.
        # Position numbers begin at padding_idx + 1.
        mask = tf.cast(tf.not_equal(input_ids, padding_idx), dtype=tf.int32)
        incremental = tf.math.cumsum(mask, axis=1)
        if past_key_values_length != 0:
            incremental = incremental + tf.cast(past_key_values_length, tf.int32)
        incremental = incremental * mask + tf.cast(padding_idx, tf.int32)
        return tf.cast(incremental, dtype=input_ids.dtype)

    def call(self, inputs: Dict[str, tf.Tensor], training=False, output_hidden_states: bool = False):
        input_ids = tf.cast(inputs["input_ids"], tf.int32)
        attention_mask = tf.cast(inputs["attention_mask"], tf.int32)

        inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        pos_ids = self._create_position_ids_from_attention_mask(attention_mask, padding_idx=self.padding_idx)
        pos_ids = tf.cast(pos_ids, dtype=input_ids.dtype)
        pos_embeds = tf.gather(params=self.position_embeddings, indices=pos_ids)

        token_type_ids = tf.zeros_like(input_ids)
        tok_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)

        hidden_states = inputs_embeds + pos_embeds + tok_type_embeds
        hidden_states = self.layerNorm(hidden_states)
        if training:
            hidden_states = self.dropout(hidden_states, training=training)

        bsz, _, _ = self._shape_list(hidden_states)
        ext_mask = tf.reshape(attention_mask, (bsz, 1, 1, -1))
        ext_mask = tf.cast(ext_mask, dtype=hidden_states.dtype)
        ext_mask = (1.0 - ext_mask) * tf.constant(-10000.0, dtype=hidden_states.dtype)

        all_h = []
        if output_hidden_states:
            all_h.append(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=ext_mask, training=training)
            if output_hidden_states:
                all_h.append(hidden_states)

        last_hidden_state = hidden_states
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * tf.cast(attention_mask[:, 1:][:, :, None], dtype=tf.float32)

        out = {"last_hidden_state": last_hidden_state, "colbert_vecs": colbert_vecs}
        if output_hidden_states:
            out["hidden_states"] = tf.stack(all_h, axis=0)
        return out
