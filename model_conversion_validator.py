import torch
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

def load_original_pytorch_model(model_name_or_path):
    """
    원본 Hugging Face(PyTorch) 모델 및 토크나이저를 로드한 뒤,
    (model, tokenizer)를 반환합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()  # 평가 모드
    return model, tokenizer


def encode_with_pytorch_model(
        model,
        tokenizer,
        queries,
        max_length=128,
        use_cls_pooling=True,
        return_hidden_states=True
):
    """
    PyTorch 모델로 임베딩 추출하는 함수.
    use_cls_pooling=True이면 [CLS] 임베딩 반환,
    False이면 Attention Mask 기반 mean pooling을 반환.
    return_hidden_states=True 이면, 모든 레이어의 히든 스테이트도 반환.
    """
    inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=return_hidden_states)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

    if use_cls_pooling:
        # [CLS] 벡터 사용
        embeddings = hidden_states[:, 0, :]
    else:
        # Mean Pooling
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

    if return_hidden_states:
        # outputs.hidden_states: 튜플 (embedding_layer_output + 각 Transformer 레이어 출력)
        all_layer_outputs = outputs.hidden_states  # tuple of torch.Tensor
        return embeddings.cpu().numpy(), all_layer_outputs
    else:
        return embeddings.cpu().numpy()


def show_all_layer_outputs_pytorch(all_layer_outputs, print_values=False):
    """
    PyTorch 레이어별 히든 스테이트의 shape 및 (옵션) 일부 실제 값을 출력하는 유틸 함수.
    """
    print("\n[PyTorch] All Layer Outputs:")
    for i, hs in enumerate(all_layer_outputs):
        print(f"  Layer {i} hidden state shape: {hs.shape}")
        if print_values:
            # 첫 배치, 첫 토큰, 앞 5개 차원
            sample_vals = hs[0, 0, :5]
            print(f"    Sample values (batch=0, token=0, dims=0~4): {sample_vals.cpu().numpy()}")
    print()


def load_converted_tf_model(saved_model_dir):
    """
    TF SavedModel 디렉토리에서 모델을 로드하고,
    같은 경로에 있는 토크나이저를 함께 로드합니다.

    - convert_and_save_model()나 save_model_with_tokenizer()로
      "model" 폴더와 토크나이저 저장 가정.
    """
    model_path = f"{saved_model_dir}/model"
    loaded_model = tf.saved_model.load(model_path)
    serving_fn = loaded_model.signatures["serving_default"]

    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
    return serving_fn, tokenizer


def encode_with_tf_model(serving_fn, tokenizer, queries, max_length=128):
    """
    TensorFlow 모델(서빙 시그니처)로 임베딩 추출하는 함수.
    BGEM3TensorFlow 구조상 "dense_vecs" 키에 최종 임베딩이 들어있다고 가정.
    """
    inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    outputs = serving_fn(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    embeddings = outputs["dense_vecs"].numpy()  # (batch_size, hidden_size)

    return embeddings


def encode_with_tf_model_and_get_hidden_states(serving_fn, tokenizer, queries, max_length=128):
    """
    *주의*:
    - TF SavedModel에서 레이어별 히든 스테이트도 반환한다고 가정할 때 사용 가능.
    - 실제 변환된 모델이 'all_hidden_states'라는 키를 노출하지 않았다면 KeyError 발생 가능.
    """
    inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    outputs = serving_fn(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    hidden_states = outputs["hidden_states"]  # (num_layers, batch, seq_len, hidden_dim)
    final_embeddings = outputs["dense_vecs"]
    print("outputs['colbert_vecs'] : ")
    print(outputs["colbert_vecs"])

    return final_embeddings.numpy(), hidden_states


def show_all_layer_outputs_tf(all_layer_outputs, print_values=False):
    """
    TensorFlow 레이어별 히든 스테이트 shape와 (옵션) 일부 실제 값을 출력
    (가정: all_layer_outputs가 (num_layers, batch, seq_len, hidden_dim) 형태)
    """
    print("\n[TensorFlow] All Layer Outputs:")
    for i, hs in enumerate(all_layer_outputs):
        print(f"  Layer {i} hidden state shape: {hs.shape}")
        if print_values:
            # 첫 배치, 첫 토큰, 앞 5개 차원
            sample_vals = hs[0, 0, :5].numpy()
            print(f"    Sample values (batch=0, token=0, dims=0~4): {sample_vals}")
    print()


def cosine_similarity(a, b):
    """
    (batch_size, hidden_dim) 형태 numpy 배열 a, b에 대해
    벡터별 코사인 유사도(batch_size,) 반환
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cos_sim = np.sum(a_norm * b_norm, axis=1)
    return cos_sim


def mse(a, b):
    return np.mean((a - b) ** 2)


def compare_layer_outputs(pt_all_layer_outputs, tf_all_layer_outputs):
   """
   PyTorch vs. TensorFlow 레이어별로 MSE, Cosine Similarity 등을 비교해주는 함수.
   - pt_all_layer_outputs: tuple of torch.Tensor (길이: num_layers_PyTorch)
     (예: [embedding_output, layer1_output, layer2_output, ...])
   - tf_all_layer_outputs: tf.Tensor (shape: [num_layers_TF, batch_size, seq_len, hidden_dim])
     (예: 0번이 embedding_output, 1번이 1번 레이어, ...)
   """
   print("\n=== Compare Layer Outputs (PyTorch vs TensorFlow) ===")

   num_pt_layers = len(pt_all_layer_outputs)
   num_tf_layers = tf_all_layer_outputs.shape[0]
   min_layers = min(num_pt_layers, num_tf_layers)


   layer_names = {
       0: "Embedding Layer",
   }
   for i in range(1, min_layers):
       layer_names[i] = f"Encoder Layer {i}"

   print("pt_all_layer_outputs", len(pt_all_layer_outputs))

   print("tf_all_layer_outputs", len(tf_all_layer_outputs))

   for layer_idx in range(min_layers):
       pt_layer = pt_all_layer_outputs[layer_idx]  # shape: [batch, seq_len, hidden_dim]
       tf_layer = tf_all_layer_outputs[layer_idx]  # shape: [batch, seq_len, hidden_dim]
       tf_layer_np = tf_layer.numpy()

       print(f"\n{layer_names[layer_idx]}:")
       print(f"\n{layer_names[layer_idx]}:")
       print(f"PyTorch shape: {pt_layer.shape}")
       print(f"    dims: [batch_size={pt_layer.shape[0]}, seq_len={pt_layer.shape[1]}, hidden_dim={pt_layer.shape[2]}]")
       print(f"TensorFlow shape: {tf_layer.shape}")
       print(f"    dims: [batch_size={tf_layer.shape[0]}, seq_len={tf_layer.shape[1]}, hidden_dim={tf_layer.shape[2]}]")

       layer_mse = mse(pt_layer.detach().cpu().numpy(), tf_layer_np)
       pt_cls_vec = pt_layer[0, 0, :].detach().cpu().numpy()

       tf_cls_vec = tf_layer_np[0, 0, :]


       cls_cos_sim = cosine_similarity(pt_cls_vec[np.newaxis, :], tf_cls_vec[np.newaxis, :])[0]

       print(f"  -> MSE: {layer_mse:.6f}")
       print(f"  -> CLS Token Cosine Similarity: {cls_cos_sim:.6f}")

# ===================== 추가한 함수: 레이어별 출력 비교 =====================
def compare_layer_outputs1(pt_all_layer_outputs, tf_all_layer_outputs):
    """
    PyTorch vs. TensorFlow 레이어별로 MSE, Cosine Similarity 등을 비교해주는 함수.
    - pt_all_layer_outputs: tuple of torch.Tensor (길이: num_layers_PyTorch)
      (예: [embedding_output, layer1_output, layer2_output, ...])
    - tf_all_layer_outputs: tf.Tensor (shape: [num_layers_TF, batch_size, seq_len, hidden_dim])
      (예: 0번이 embedding_output, 1번이 1번 레이어, ...)
    """
    print("\n=== Compare Layer Outputs (PyTorch vs TensorFlow) ===")

    # PyTorch: len(pt_all_layer_outputs) = num_layers_PyTorch
    # TensorFlow: tf_all_layer_outputs.shape[0] = num_layers_TF
    num_pt_layers = len(pt_all_layer_outputs)
    num_tf_layers = tf_all_layer_outputs.shape[0]

    # 두 모델 간 레이어 개수가 다를 수 있으므로, 비교 가능한 만큼만 비교
    min_layers = min(num_pt_layers, num_tf_layers)

    for layer_idx in range(min_layers):
        pt_layer = pt_all_layer_outputs[layer_idx]  # shape: [batch, seq_len, hidden_dim]
        tf_layer = tf_all_layer_outputs[layer_idx]  # shape: [batch, seq_len, hidden_dim]
        tf_layer_np = tf_layer.numpy()

        # 일단 shape이 같은지 출력
        print(f"Layer {layer_idx}: PT {pt_layer.shape} vs TF {tf_layer.shape}")

        # MSE 계산
        layer_mse = mse(pt_layer.detach().cpu().numpy(), tf_layer_np)
        # Cosine Sim: 여기서는 batch*seq_len 개 각 토큰별 벡터의 평균 코사인 유사도 등
        # 또는 첫 배치의 첫 토큰만 비교할 수도 있음
        # 여기서는 간단히 "CLS 토큰(즉 0번 token)에 대한 cos sim" 등 비교 예시
        pt_cls_vec = pt_layer[0, 0, :].detach().cpu().numpy()
        tf_cls_vec = tf_layer_np[0, 0, :]

        print(pt_layer)
        print(tf_layer_np)
        cls_cos_sim = cosine_similarity(pt_cls_vec[np.newaxis, :], tf_cls_vec[np.newaxis, :])[0]

        print(f"  -> MSE: {layer_mse:.6f},  CLS CosSim: {cls_cos_sim:.6f}")
    print()


def main():
    # 경로 설정 (예: ./bge-m3, ./converted_bge_m3)
    model_name_or_path = "./bge-m3"  # PyTorch 원본
    saved_model_dir = "./converted_bge_m3"  # TF 변환본

    queries = [
        "이 모델은 무엇을 하는 모델인가요?",
        "이 모델은 무엇을 하는 모델인가요?"
    ]

    print("=== 1) PyTorch 모델 로드 및 인코딩 (레이어별 출력 포함) ===")
    pt_model, pt_tokenizer = load_original_pytorch_model(model_name_or_path)
    pt_embeddings, pt_all_layer_outputs = encode_with_pytorch_model(
        pt_model,
        pt_tokenizer,
        queries,
        max_length=128,
        use_cls_pooling=True,
        return_hidden_states=True
    )
    show_all_layer_outputs_pytorch(pt_all_layer_outputs, print_values=False)

    print("=== 2) TensorFlow 모델 로드 및 인코딩 ===")
    tf_serving_fn, tf_tokenizer = load_converted_tf_model(saved_model_dir)
    tf_embeddings = encode_with_tf_model(
        tf_serving_fn,
        tf_tokenizer,
        queries,
        max_length=128
    )

    # (옵션) 레이어별 출력 노출 여부 확인
    try:
        tf_embeddings_with_layers, tf_all_layer_outputs = encode_with_tf_model_and_get_hidden_states(
            tf_serving_fn,
            tf_tokenizer,
            queries,
            max_length=128
        )
        show_all_layer_outputs_tf(tf_all_layer_outputs, print_values=False)

        # [추가] 레이어별로 직접 비교
        compare_layer_outputs(pt_all_layer_outputs, tf_all_layer_outputs)

        print("[TensorFlow] Final Embeddings Shape:", tf_embeddings_with_layers.shape)
    except KeyError:
        print("TensorFlow 서빙 시그니처에 hidden_states가 없습니다. (기본 TF 변환본일 가능성)")

    print("\n=== 3) PT vs. TF 최종 임베딩 비교 ===")

    print(pt_embeddings)
    print(tf_embeddings)

    cos_sims = cosine_similarity(pt_embeddings, tf_embeddings)

    errors = (pt_embeddings - tf_embeddings)
    mse_val = mse(pt_embeddings, tf_embeddings)

    print("===== Queries =====")
    for i, q in enumerate(queries):
        print(f"[{i}] {q}")
    print()

    print("===== PyTorch Embeddings (shape) =====")
    print(pt_embeddings.shape)
    print("===== TF Embeddings (shape) =====")
    print(tf_embeddings.shape)

    print("\n===== Pairwise Cosine Similarity (PT vs TF) =====")
    for i, cs in enumerate(cos_sims):
        print(f"Query {i} Cosine Similarity: {cs:.4f}")

    print(f"\n===== MSE (PT vs TF) =====")
    print(f"MSE: {mse_val:.6f}")

    print("\n===== Sample Differences (first query, first 5 dims) =====")
    print(errors[0][:5])


if __name__ == "__main__":
    main()
