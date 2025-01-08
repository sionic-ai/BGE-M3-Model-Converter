from BGEM3TFModel import BGEM3TensorFlow, save_model_with_tokenizer

from transformers import AutoModel
import numpy as np
import tensorflow as tf

import os
import torch


def load_sparse_weights():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, './bge-m3', 'sparse_linear.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FileNotFoundError: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(model_path, map_location=device, weights_only=True)


def load_colbert_weights():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, './bge-m3', 'colbert_linear.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FileNotFoundError: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(model_path, map_location=device, weights_only=True)


def _init_colbert_weights(tf_model):
    """Initialize pooler weights"""

    colbert = load_colbert_weights()
    colbert_weights = colbert['weight']
    colbert_bias = colbert['bias']

    tf_model.colbert_linear.set_weights([
        colbert_weights.numpy().T,
        colbert_bias.numpy()
    ])


class BGEM3WeightConverter:
    def __init__(self, model_name_or_path: str):
        self.pt_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.pt_state_dict = self.pt_model.state_dict()

        for name, param in self.pt_model.state_dict().items():
            print(f"{name:<30} | shape: {param.shape}")

    def initialize_weights(self, tf_model):
        """Initialize TensorFlow model with PyTorch weights"""
        # Build model with dummy inputs first
        dummy_inputs = {
            'input_ids': tf.zeros((2, 12), dtype=tf.int32),
            'attention_mask': tf.ones((2, 12), dtype=tf.int32)
        }
        _ = tf_model(dummy_inputs, training=False)

        # Initialize embeddings
        self._init_embedding_weights(tf_model)

        # Initialize encoder layers
        self._init_transformer_blocks(tf_model)

        # Initialize pooler
        self._init_pooler_weights(tf_model)

        # Initialize pooler
        self._init_pooler_weights(tf_model)

        # Initialize colbert
        _init_colbert_weights(tf_model)

        return tf_model

    def _init_embedding_weights(self, tf_model):
        """Initialize embedding layer weights"""

        # Word embeddings
        tf_model.weight.assign(
            self.pt_state_dict['embeddings.word_embeddings.weight'].numpy()
        )

        # Position embeddings
        tf_model.position_embeddings.assign(
            self.pt_state_dict['embeddings.position_embeddings.weight'].numpy()
        )

        # token type embeddings
        tf_model.token_type_embeddings.assign(
            self.pt_state_dict['embeddings.token_type_embeddings.weight'].numpy()
        )

        # Layer normalization
        tf_model.layerNorm.set_weights([
            self.pt_state_dict['embeddings.LayerNorm.weight'].numpy(),
            self.pt_state_dict['embeddings.LayerNorm.bias'].numpy()
        ])

    def _init_transformer_blocks(self, tf_model):
        """Initialize transformer block weights"""
        for i, layer in enumerate(tf_model.encoder_layers):
            prefix = f'encoder.layer.{i}.'

            # Attention weights
            self._init_attention_weights(layer, prefix, tf_model)

            # Feed-forward weights
            self._init_ffn_weights(layer, prefix)

    def _init_attention_weights(self, layer, prefix, tf_model):
        """Initialize multi-head attention weights with correct reshaping"""
        # Load weights from PyTorch
        q_weight = self.pt_state_dict[f'{prefix}attention.self.query.weight'].numpy()
        k_weight = self.pt_state_dict[f'{prefix}attention.self.key.weight'].numpy()
        v_weight = self.pt_state_dict[f'{prefix}attention.self.value.weight'].numpy()

        q_bias = self.pt_state_dict[f'{prefix}attention.self.query.bias'].numpy()
        k_bias = self.pt_state_dict[f'{prefix}attention.self.key.bias'].numpy()
        v_bias = self.pt_state_dict[f'{prefix}attention.self.value.bias'].numpy()

        out_weight = self.pt_state_dict[f'{prefix}attention.output.dense.weight'].numpy()
        out_bias = self.pt_state_dict[f'{prefix}attention.output.dense.bias'].numpy()

        output_lm_weight = self.pt_state_dict[f'{prefix}attention.output.LayerNorm.weight'].numpy()
        output_lm_bias = self.pt_state_dict[f'{prefix}attention.output.LayerNorm.bias'].numpy()

        # Set weights in the correct order for TensorFlow
        layer.attention.wq.set_weights([
            q_weight.T,  # (num_heads, hidden_size, head_size)
            q_bias,  # (num_heads, head_size)
        ])

        layer.attention.wk.set_weights([
            k_weight.T,  # (num_heads, hidden_size, head_size)
            k_bias,  # (num_heads, head_size)
        ])

        layer.attention.wv.set_weights([
            v_weight.T,  # (num_heads, hidden_size, head_size)
            v_bias,  # (`num_heads, head_size)
        ])

        layer.attention.dense.set_weights([
            out_weight.T,  # (num_heads, head_size, hidden_size)
            out_bias  # (hidden_size,)
        ])

        # Layer normalization
        layer.attention.attlayerNorm.set_weights([
            output_lm_weight,
            output_lm_bias
        ])

        # Origin shape 출력
        print(f"\nOriginal weights shape for {prefix}:")
        print(f"Query weight: {q_weight.shape}")

    def _init_ffn_weights(self, layer, prefix):
        """Initialize feed-forward network weights"""

        # Intermediate dense
        layer.intermediate.set_weights([
            self.pt_state_dict[f'{prefix}intermediate.dense.weight'].numpy().T,
            self.pt_state_dict[f'{prefix}intermediate.dense.bias'].numpy()
        ])

        # Output dense
        layer.output_dense.set_weights([
            self.pt_state_dict[f'{prefix}output.dense.weight'].numpy().T,
            self.pt_state_dict[f'{prefix}output.dense.bias'].numpy()
        ])

        # Layer norm
        layer.output_norm.set_weights([
            self.pt_state_dict[f'{prefix}output.LayerNorm.weight'].numpy(),
            self.pt_state_dict[f'{prefix}output.LayerNorm.bias'].numpy()
        ])

    def _init_pooler_weights(self, tf_model):
        """Initialize pooler weights"""

        pooler_weight = self.pt_state_dict['pooler.dense.weight'].numpy()
        pooler_bias = self.pt_state_dict['pooler.dense.bias'].numpy()

        print("\nPooler weights debug info:")
        print(f"PyTorch pooler weight shape: {pooler_weight.shape}")
        print(f"PyTorch pooler weight stats: mean={np.mean(pooler_weight):.6f}, std={np.std(pooler_weight):.6f}")
        print(f"PyTorch pooler bias stats: mean={np.mean(pooler_bias):.6f}, std={np.std(pooler_bias):.6f}")

        # 가중치 설정 전에 dummy forward pass로 build
        dummy_input = tf.random.normal([1, 10, pooler_weight.shape[1]])
        _ = tf_model.pooler(dummy_input[:, 0, :])  # [CLS] 토큰만 사용하도록 수정

        # 가중치 설정
        tf_model.pooler.set_weights([
            pooler_weight,
            pooler_bias
        ])

        # 검증
        pooler_output = tf_model.pooler(dummy_input[:, 0, :])
        print(f"\nTF Pooler test output stats:")
        print(f"shape: {pooler_output.shape}")
        print(f"mean: {tf.reduce_mean(pooler_output):.6f}")
        print(f"std: {tf.math.reduce_std(pooler_output):.6f}")


def convert_and_save_model(model_name: str, save_path: str):
    """Convert PyTorch model to TensorFlow and save"""
    # Initialize TensorFlow model
    tf_model = BGEM3TensorFlow(model_name)

    # Convert weights
    converter = BGEM3WeightConverter(model_name)
    tf_model = converter.initialize_weights(tf_model)

    # Save model
    tokenizer = tf_model.tokenizer
    save_model_with_tokenizer(tf_model, tokenizer, save_path)

    return tf_model


if __name__ == "__main__":
    model_name = "BAAI/bge-m3"
    save_path = "./converted_bge_m3"

    tf_model = convert_and_save_model(model_name, save_path)
    print("Model converted and saved successfully!")
