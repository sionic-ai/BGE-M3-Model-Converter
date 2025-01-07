import os
from transformers import AutoTokenizer, AutoConfig

import tensorflow as tf
import math
from typing import List, Union

import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads  # 각 헤드의 차원 크기

        # Query, Key, Value를 위한 Dense Layer
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # 출력 레이어
        self.dense = tf.keras.layers.Dense(d_model)

        # 어텐션 layerNorm
        self.attlayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # 드롭아웃
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def stable_softmax(self, logits, axis=None, name=None):
        """
        Stable softmax implementation
        """
        return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, inputs, mask=None, training=False):
        batch_size = tf.shape(inputs)[0]

        # Query, Key, Value를 계산
        q = self.wq(inputs)  # (batch_size, seq_len, d_model)
        k = self.wk(inputs)  # (batch_size, seq_len, d_model)
        v = self.wv(inputs)  # (batch_size, seq_len, d_model)

        # 다중 헤드로 분리
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled Dot-Product Attention
        sqrt_att_head_size = math.sqrt(self.depth)

        attention_scores = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(sqrt_att_head_size, tf.float32)
        attention_scores = tf.divide(attention_scores, dk)

        attention_probs = self.stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        # Attention result
        attention_output = tf.matmul(attention_probs, v)  # (batch_size, num_heads, seq_len_q, depth)
        attention_output = tf.transpose(attention_output,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        attention_output = tf.reshape(attention_output,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # Dense Layer
        output = self.dense(attention_output)  # (batch_size, seq_len_q, d_model)
        if training:
            output = self.dropout(inputs=output, training=training)
        # Residual
        output = self.attlayerNorm(inputs=output + inputs)

        return output, attention_probs


class BGEM3TensorFlow(tf.keras.Model):
    def __init__(self, model_name, normalize_embeddings=False, use_fp16=True,
                 query_instruction_for_retrieval=None, query_instruction_format="{}{}",
                 pooling_method="cls", trust_remote_code=False, cache_dir=None,
                 colbert_dim=-1, batch_size=256, query_max_length=512,
                 passage_max_length=512, return_dense=True, return_sparse=False,
                 return_colbert_vecs=False, dropout_rate=0.1):
        super().__init__(name="bge-m3-tensorflow")

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        self.dropout_rate = dropout_rate
        self.padding_idx = 1

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # Model hyperparameters
        self.d_model = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_layers = self.config.num_hidden_layers
        self.vocab_size = self.config.vocab_size

        # Build components
        self._build_embeddings()
        self._build_encoder_layers()
        self._build_pooler()
        self._build_colbert()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )

    def shape_list(self, tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
        if isinstance(tensor, np.ndarray):
            return list(tensor.shape)

        dynamic = tf.shape(tensor)

        if tensor.shape == tf.TensorShape(None):
            return dynamic

        static = tensor.shape.as_list()

        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0, padding_idx=1):
        mask = tf.cast(tf.math.not_equal(input_ids, padding_idx), dtype=input_ids.dtype)
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask
        return incremental_indices + padding_idx

    def _build_embeddings(self):
        """Build embedding layers following XLMRoberta structure"""
        # Word embeddings
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="embeddings",
                shape=[self.vocab_size, self.d_model],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            )

        # Position embeddings
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.max_position_embeddings, self.d_model],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            )

        # Token type embeddings
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.d_model],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            )

        # LayerNorm
        self.layerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.config.layer_norm_eps,
            name="LayerNorm"
        )

        # Dropout
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def _build_encoder_layers(self):
        """Build transformer encoder layers"""
        self.encoder_layers = []
        for i in range(self.num_layers):
            layer = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.dropout_rate,
                name=f"encoder.layer.{i}"
            )
            self.encoder_layers.append(layer)

    def _build_pooler(self):
        """Build pooler layer"""
        self.pooler = tf.keras.layers.Dense(
            self.d_model,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="pooler.dense"
        )

    def _build_colbert(self):
        self.colbert_linear = tf.keras.layers.Dense(
            units=self.d_model,
        )

    def call(self, inputs, training=False, output_hidden_states=False):

        input_ids = tf.cast(inputs['input_ids'], tf.int32)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)
        position_ids = inputs.get("position_ids", None)

        input_shape = self.shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids=input_ids,
                                                                       padding_idx=self.padding_idx)
            else:
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)

        # Sum embeddings
        embedding_output = inputs_embeds + position_embeds + token_type_embeds

        # Apply LayerNorm and Dropout
        embedding_output = self.layerNorm(embedding_output)
        if training:
            embedding_output = self.dropout(embedding_output, training=training)

        attention_mask_origin = attention_mask

        attention_mask_shape = self.shape_list(attention_mask)

        extended_attention_mask = tf.reshape(
            attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
        )

        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        attention_mask = extended_attention_mask

        all_hidden_states = [embedding_output] if output_hidden_states else []

        hidden_states = embedding_output
        # Pass through encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Pooling
        if self.pooling_method == "mean":
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
        else:  # default: cls
            pooled_output = hidden_states[:, 0, :]

        # Apply pooler if return_dense is True
        if self.return_dense:
            pooled_output = pooled_output

        # Normalize embeddings if specified
        if self.normalize_embeddings:
            pooled_output = tf.nn.l2_normalize(pooled_output, axis=-1)

        ## colbert_vecs
        colbert_vecs = self.colbert_linear(hidden_states[:, 1:])
        colbert_vecs = colbert_vecs * tf.cast(attention_mask_origin[:, 1:][:, :, None], dtype=tf.float32)

        outputs = {
            "dense_vecs": pooled_output,
            "colbert_vecs": colbert_vecs,
            "last_hidden_state": hidden_states
        }

        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states

        return outputs


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, intermediate_size, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Intermediate -> gelu_approx
        self.intermediate = tf.keras.layers.Dense(
            intermediate_size,
            name="intermediate.dense"
        )
        self.output_dense = tf.keras.layers.Dense(d_model, name="output.dense")
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def gelu_approx(self, x):
        x = tf.convert_to_tensor(x)
        cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))

        return x * cdf

    def call(self, x, attention_mask=None, training=False):
        # Self-attention
        attention_output, attention_weights = self.attention(
            inputs=x,
            mask=attention_mask,
            training=training
        )

        # FFN with GELU activation
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.gelu_approx(intermediate_output)  # Use exact GELU approximation
        layer_output = self.output_dense(intermediate_output)
        if training:
            layer_output = self.output_dropout(layer_output, training=training)
        # Residual
        output = layer_output + attention_output
        output = self.output_norm(output)

        return output


def save_model_with_tokenizer(model, tokenizer, save_path):
    """Save both model and tokenizer"""
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, 'model')

    # Ensure model is built by calling it with dummy inputs
    dummy_inputs = {
        'input_ids': tf.zeros((2, 11), dtype=tf.int32),
        'attention_mask': tf.ones((2, 11), dtype=tf.int32)
    }
    _ = model(dummy_inputs, training=False, output_hidden_states=True)

    # Define serving signature
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_ids'),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='attention_mask')
    ])
    def serving_fn(input_ids, attention_mask):

        print(input_ids)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        outputs = model(inputs=inputs, training=False, output_hidden_states=True)

        if outputs.get('hidden_states'):
            hidden_states = tf.stack(outputs['hidden_states'], axis=0)
            return {
                'dense_vecs': outputs['dense_vecs'],  # CLS Token
                'colbert_vecs': outputs['colbert_vecs'],
                'hidden_states': hidden_states  # (num_layers, batch, seq_len, hidden_dim)
            }
        else:
            return {
                'dense_vecs': outputs['dense_vecs'],
            }

    # Save model
    tf.saved_model.save(
        model,
        model_save_path,
        signatures={'serving_default': serving_fn}
    )

    # Save tokenizer
    tokenizer.save_pretrained(save_path)

    return model_save_path
