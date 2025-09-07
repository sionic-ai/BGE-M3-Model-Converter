# model_conversion_validator.py
import numpy as np
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel


def load_original_pytorch_model(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModel.from_pretrained(model_name_or_path)
    mdl.eval()
    return mdl, tok


def encode_with_pytorch_model(model, tokenizer, queries, max_length=128, use_cls_pooling=True):
    inputs = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # [B,T,H]
        all_layer_outputs = outputs.hidden_states  # tuple(len=emb+24)
    if use_cls_pooling:
        emb = hidden_states[:, 0, :].cpu().numpy()
    else:
        attn = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * attn, dim=1)
        sum_mask = torch.clamp(attn.sum(dim=1), min=1e-9)
        emb = (sum_embeddings / sum_mask).cpu().numpy()


    return emb, all_layer_outputs


def show_all_layer_outputs_pytorch(all_layer_outputs, print_values=False):
    print("\n[PyTorch] All Layer Outputs:")
    for i, hs in enumerate(all_layer_outputs):
        print(f"  Layer {i} hidden state shape: {tuple(hs.shape)}")
        if print_values:
            print("    sample:", hs[0, 0, :5].cpu().numpy())


def load_converted_tf_model(saved_root_dir: str):
    model_dir = f"{saved_root_dir}/model"
    loaded = tf.saved_model.load(model_dir)
    sig = loaded.signatures["serving_default"]
    tok = AutoTokenizer.from_pretrained(saved_root_dir)
    return sig, tok


def call_signature(sig, input_ids, attention_mask):
    # 강제 int32 캐스트
    if input_ids.dtype != tf.int32:
        input_ids = tf.cast(input_ids, tf.int32)
    if attention_mask.dtype != tf.int32:
        attention_mask = tf.cast(attention_mask, tf.int32)
    return sig(input_ids=input_ids, attention_mask=attention_mask)


def encode_with_tf_model(serving_fn, tokenizer, queries, max_length=128):
    inputs_pt = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs_tf = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="tf")

    # 1) 입력 동일성 보장 (매우 중요)
    assert np.array_equal(inputs_pt["input_ids"].numpy(), inputs_tf["input_ids"].numpy()), "PT/TF input_ids mismatch"
    assert np.array_equal(inputs_pt["attention_mask"].numpy(), inputs_tf["attention_mask"].numpy()), "PT/TF mask mismatch"

    outputs = serving_fn(
        input_ids=tf.cast(inputs_tf["input_ids"], tf.int32),
        attention_mask=tf.cast(inputs_tf["attention_mask"], tf.int32),
    )
    print(f'outputs >> {outputs}')
    last_hidden = outputs["last_hidden_state"]    # [B,T,H]
    emb = last_hidden[:, 0, :].numpy()
    hiddens = outputs.get("hidden_states", None)  # (L+1,B,T,H)
    print(f'hiddens, {hiddens}')
    return emb, (hiddens.numpy() if hiddens is not None else None)


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a * b, axis=1)


def mse(a, b):
    return np.mean((a - b) ** 2)


def manual_l0_from_pt(sd, input_ids_np, attention_mask_np, padding_idx=1, eps=1e-5):
    we = sd["embeddings.word_embeddings.weight"].cpu().numpy().astype(np.float32)
    pe = sd["embeddings.position_embeddings.weight"].cpu().numpy().astype(np.float32)
    te = sd["embeddings.token_type_embeddings.weight"].cpu().numpy().astype(np.float32)
    gamma = sd["embeddings.LayerNorm.weight"].cpu().numpy().astype(np.float32)
    beta  = sd["embeddings.LayerNorm.bias"].cpu().numpy().astype(np.float32)

    # HF와 동일: attention_mask로 포지션 ID 생성
    mask = attention_mask_np.astype(np.int32)
    pos_ids = np.cumsum(mask, axis=1) * mask + padding_idx

    emb = we[input_ids_np] + pe[pos_ids] + te[0]  # type_vocab_size == 1
    mean = emb.mean(axis=-1, keepdims=True)
    var  = ((emb - mean) ** 2).mean(axis=-1, keepdims=True)  # 모집단 분산
    xhat = (emb - mean) / np.sqrt(var + eps)
    return xhat * gamma + beta  # (B,T,H)




def main():
    pt_id = "BAAI/bge-m3"
    tf_dir = "./converted_bge_m3_tf1_java_fixed"

    queries = [
        "이 모델은 무엇을 하는 모델인가요? 이 모델은 무엇을 하는 모델인가요?",
        "이 모델은 무엇을 하는 모델인가요?"
    ]

    print("=== 1) PyTorch ===")
    pt_model, pt_tok = load_original_pytorch_model(pt_id)
    pt_emb, pt_layers = encode_with_pytorch_model(pt_model, pt_tok, queries, max_length=128)
    show_all_layer_outputs_pytorch(pt_layers)

    print("=== 2) TensorFlow ===")
    tf_sig, tf_tok = load_converted_tf_model(tf_dir)
    tf_emb, tf_layers = encode_with_tf_model(tf_sig, tf_tok, queries, max_length=128)

    pt_l0 = pt_layers[0].detach().cpu().numpy()  # (B,T,H)
    tf_l0 = tf_layers[0]  # (B,T,H)
    print("L0 CLS head(PT)[:8]:", pt_l0[0, 0, :8])
    print("L0 CLS head(TF)[:8]:", tf_l0[0, 0, :8])

    print("\n=== 3) Compare ===")
    print("PT shape:", pt_emb.shape, "TF shape:", tf_emb.shape)
    cs = cosine_similarity(pt_emb, tf_emb)
    print("Cosine:", ["%.4f" % c for c in cs])
    print("MSE:", float(mse(pt_emb, tf_emb)))

    # 선택: 레이어별 비교 (있을 때만)
    print(f'tf_layers, {tf_layers}')
    if tf_layers is not None:
        print("\n[Layer-wise] Cosine (PT vs TF):")
        # pt_layers: tuple(len=L+1), tf_layers: (L+1,B,T,H)
        tf_layers_np = tf_layers  # (L+1,B,T,H)
        for i in range(len(pt_layers)):
            pt_l = pt_layers[i].detach().cpu().numpy()
            tf_l = tf_layers_np[i]
            c = cosine_similarity(pt_l[:, 0, :], tf_l[:, 0, :])  # CLS만 비교
            e = mse(pt_l, tf_l)
            print(f"  Layer {i:02d}  cos={c.mean():.4f}  mse={e:.6f}")

    from transformers import AutoModel
    pt_model = AutoModel.from_pretrained("BAAI/bge-m3")
    sd = pt_model.state_dict()
    inputs_pt = pt_tok(queries, padding=True, truncation=True, max_length=128, return_tensors="pt")

    l0_manual = manual_l0_from_pt(sd,
                                  inputs_pt["input_ids"].numpy(),
                                  inputs_pt["attention_mask"].numpy(),
                                  padding_idx=1,
                                  eps=float(pt_model.config.layer_norm_eps))

    pt_l0 = pt_layers[0].detach().cpu().numpy()
    tf_l0 = tf_layers[0]

    print("Manual vs PT  MSE:", np.mean((l0_manual - pt_l0) ** 2))
    print("Manual vs TF  MSE:", np.mean((l0_manual - tf_l0) ** 2))


if __name__ == "__main__":
    main()
