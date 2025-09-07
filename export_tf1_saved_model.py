import os
import argparse
import numpy as np
import torch
import tensorflow as tf
from transformers import AutoTokenizer
# from BGEM3TFModel_tfkeras2 import BGEM3TensorFlow

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

def _np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    arr = np.array(t)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

def load_state_dict(model_or_path: str) -> dict:
    if os.path.isdir(model_or_path):
        pt = os.path.join(model_or_path, "pytorch_model.bin")
    else:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=model_or_path)
        pt = os.path.join(local, "pytorch_model.bin")
    print(f"[load] pytorch_model.bin: {pt}")
    return torch.load(pt, map_location="cpu")

def load_colbert_weight(model_or_path: str):
    try:
        if os.path.isdir(model_or_path):
            p = os.path.join(model_or_path, "colbert_linear.pt")
        else:
            from huggingface_hub import snapshot_download
            local = snapshot_download(repo_id=model_or_path)
            p = os.path.join(local, "colbert_linear.pt")
        st = torch.load(p, map_location="cpu")
        if isinstance(st, dict):
            W = st.get("weight")
            B = st.get("bias")
            if W is None:
                first_key = next(iter(st))
                W = st[first_key]
                B = st.get("bias", None)
        else:
            W, B = (st, None) if not isinstance(st, (list, tuple)) else st
        W = _np(W)
        B = _np(B) if B is not None else None
        print(f"[colbert] head: out_dim={W.shape[0]}, in_dim={W.shape[1]}, bias={'yes' if B is not None else 'no'}")
        return W, B
    except Exception as e:
        print(f"[colbert] not found ({e}); skipping)")
        return None, None

def _to_tf_dtype(d) -> tf.dtypes.DType:
    try:
        return tf.as_dtype(d)
    except Exception:
        if hasattr(d, "dtype"):
            try:
                return tf.as_dtype(d.dtype)
            except Exception:
                pass
        if isinstance(d, str):
            return tf.as_dtype(d)
        return tf.float32

def assign_via_feed(sess: tf.compat.v1.Session, var, value_np: np.ndarray):
    """
    그래프에 대용량 Const를 남기지 않도록 placeholder feed 기반으로 assign.
    var.dtype가 문자열("float32")이어도 안전하게 동작하도록 tf.as_dtype로 강제변환.
    """
    # KerasVariable -> tf.Variable 강제 변환 (그래프 모드에서 안전)
    var = _as_tf_variable(var)
    # var.dtype이 'float32' 같은 문자열일 수 있으므로 반드시 캐스팅
    dtype = tf.as_dtype(getattr(var, "dtype", tf.float32))
    try:
        base_dtype = dtype.base_dtype
    except Exception:
        base_dtype = dtype

    ph = tf.compat.v1.placeholder(
        dtype=base_dtype,
        shape=value_np.shape,
        name=var.name.split(":")[0] + "_ph",
    )
    # 그래프 모드 assign을 명시적으로 사용
    op = tf.compat.v1.assign(var, ph)
    sess.run(op, feed_dict={ph: value_np})

def _force_build_dense(layer: tf.keras.layers.Dense, hidden_size: int):
    dummy = tf.zeros([1, hidden_size], dtype=tf.float32)
    _ = layer(dummy)

def _debug_dump_embedding_stats(sess, tf_model):
    """
    TF1 그래프 모드에서 Keras 3 변수를 안전하게 읽어서 통계를 출력.
    KerasVariable -> (var.value) -> (read_value()) -> Tensor -> sess.run()
    """
    def _eval(sess, var_like):
        v = _as_tf_variable(var_like)
        # 최종적으로 Variable/Tensor를 fetch
        return sess.run(v)

    w = _eval(sess, tf_model.weight)
    p = _eval(sess, tf_model.position_embeddings)
    t = _eval(sess, tf_model.token_type_embeddings)
    g = _eval(sess, tf_model.layerNorm.gamma)
    b = _eval(sess, tf_model.layerNorm.beta)

    print(f"[check] word_emb  mean={w.mean():.6f} std={w.std():.6f}")
    print(f"[check] pos_emb   mean={p.mean():.6f} std={p.std():.6f}")
    print(f"[check] tok_emb   mean={t.mean():.6f} std={t.std():.6f}")
    print(f"[check] emb_LN γ  mean={g.mean():.6f} std={g.std():.6f}")
    print(f"[check] emb_LN β  mean={b.mean():.6f} std={b.std():.6f}")


def _pt_style_l0(sd, input_ids_np, attention_mask_np, padding_idx=1, eps=1e-5):
    """PyTorch 수식과 동일한 방식으로 임베딩+LayerNorm(L0) 계산."""
    we = _np(sd["embeddings.word_embeddings.weight"])
    pe = _np(sd["embeddings.position_embeddings.weight"])
    te = _np(sd["embeddings.token_type_embeddings.weight"])  # [type_vocab_size, H]
    gamma = _np(sd["embeddings.LayerNorm.weight"])
    beta  = _np(sd["embeddings.LayerNorm.bias"])

    mask = attention_mask_np.astype(np.int32)
    pos_ids = np.cumsum(mask, axis=1) * mask + padding_idx

    emb = we[input_ids_np] + pe[pos_ids] + te[0]
    mean = emb.mean(axis=-1, keepdims=True)
    var  = ((emb - mean) ** 2).mean(axis=-1, keepdims=True)
    xhat = (emb - mean) / np.sqrt(var + eps)
    return xhat * gamma + beta

def export_tf1_saved_model(model_name_or_path: str, out_root: str):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    print("[tokenizer] loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print("[state_dict] loading...")
    sd = load_state_dict(model_name_or_path)
    col_W, col_b = load_colbert_weight(model_name_or_path)

    g = tf.compat.v1.Graph()
    with g.as_default():
        # Placeholders
        input_ids          = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="input_ids")
        attention_mask     = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="attention_mask")
        input_ids_i64      = tf.compat.v1.placeholder(tf.int64, shape=[None, None], name="input_ids_i64")
        attention_mask_i64 = tf.compat.v1.placeholder(tf.int64, shape=[None, None], name="attention_mask_i64")

        hidden_size = sd["embeddings.word_embeddings.weight"].shape[1]
        tf_model = BGEM3TensorFlow(
            vocab_size=sd["embeddings.word_embeddings.weight"].shape[0],
            max_position_embeddings=sd["embeddings.position_embeddings.weight"].shape[0],
            type_vocab_size=sd["embeddings.token_type_embeddings.weight"].shape[0],
            hidden_size=hidden_size,
            num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
            dropout_rate=0.0, name="bge-m3-tensorflow",
        )

        # 구현 타입 확인: 모두 TorchLayerNormTF1 이어야 함
        try:
            print("[impl] embeddings LN:", type(tf_model.layerNorm))
            print("[impl] attn LN(0):", type(tf_model.encoder_layers[0].attention.attlayerNorm))
            print("[impl] ffn  LN(0):", type(tf_model.encoder_layers[0].output_norm))
        except Exception:
            pass

        # Build graph (variables created here)
        outs32 = tf_model({"input_ids": input_ids, "attention_mask": attention_mask},
                          training=False, output_hidden_states=True)
        last32  = tf.identity(outs32["last_hidden_state"], name="last_hidden_state")
        col32   = tf.identity(outs32["colbert_vecs"],     name="colbert_vecs")
        dense32 = tf.identity(last32[:, 0, :],            name="dense_vecs")      # raw CLS
        hidd32  = tf.identity(outs32["hidden_states"],    name="hidden_states")

        outs64 = tf_model({"input_ids": tf.cast(input_ids_i64, tf.int32),
                           "attention_mask": tf.cast(attention_mask_i64, tf.int32)},
                          training=False, output_hidden_states=True)
        last64  = tf.identity(outs64["last_hidden_state"], name="last_hidden_state_i64")
        col64   = tf.identity(outs64["colbert_vecs"],     name="colbert_vecs_i64")
        dense64 = tf.identity(last64[:, 0, :],            name="dense_vecs_i64")  # raw CLS
        hidd64  = tf.identity(outs64["hidden_states"],    name="hidden_states_i64")

        # pooler 강제 빌드(검증용)
        _force_build_dense(tf_model.pooler, hidden_size)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session(graph=g) as sess:
            sess.run(init)

            # === Embedding block ===
            assign_via_feed(sess, tf_model.weight,                _np(sd["embeddings.word_embeddings.weight"]))
            assign_via_feed(sess, tf_model.position_embeddings,   _np(sd["embeddings.position_embeddings.weight"]))
            assign_via_feed(sess, tf_model.token_type_embeddings, _np(sd["embeddings.token_type_embeddings.weight"]))
            assign_via_feed(sess, tf_model.layerNorm.gamma,       _np(sd["embeddings.LayerNorm.weight"]))
            assign_via_feed(sess, tf_model.layerNorm.beta,        _np(sd["embeddings.LayerNorm.bias"]))

            # 즉시 숫자 확인 (여기서 이상하면 레이어 0부터 틀어집니다)
            _debug_dump_embedding_stats(sess, tf_model)

            # === Encoder blocks ===
            for i, blk in enumerate(tf_model.encoder_layers):
                assign_via_feed(sess, blk.attention.wq.kernel, _np(sd[f"encoder.layer.{i}.attention.self.query.weight"]).T)
                assign_via_feed(sess, blk.attention.wq.bias,   _np(sd[f"encoder.layer.{i}.attention.self.query.bias"]))
                assign_via_feed(sess, blk.attention.wk.kernel, _np(sd[f"encoder.layer.{i}.attention.self.key.weight"]).T)
                assign_via_feed(sess, blk.attention.wk.bias,   _np(sd[f"encoder.layer.{i}.attention.self.key.bias"]))
                assign_via_feed(sess, blk.attention.wv.kernel, _np(sd[f"encoder.layer.{i}.attention.self.value.weight"]).T)
                assign_via_feed(sess, blk.attention.wv.bias,   _np(sd[f"encoder.layer.{i}.attention.self.value.bias"]))

                assign_via_feed(sess, blk.attention.dense.kernel, _np(sd[f"encoder.layer.{i}.attention.output.dense.weight"]).T)
                assign_via_feed(sess, blk.attention.dense.bias,   _np(sd[f"encoder.layer.{i}.attention.output.dense.bias"]))
                assign_via_feed(sess, blk.attention.attlayerNorm.gamma,
                                _np(sd[f"encoder.layer.{i}.attention.output.LayerNorm.weight"]))
                assign_via_feed(sess, blk.attention.attlayerNorm.beta,
                                _np(sd[f"encoder.layer.{i}.attention.output.LayerNorm.bias"]))

                assign_via_feed(sess, blk.intermediate.kernel, _np(sd[f"encoder.layer.{i}.intermediate.dense.weight"]).T)
                assign_via_feed(sess, blk.intermediate.bias,   _np(sd[f"encoder.layer.{i}.intermediate.dense.bias"]))
                assign_via_feed(sess, blk.output_dense.kernel, _np(sd[f"encoder.layer.{i}.output.dense.weight"]).T)
                assign_via_feed(sess, blk.output_dense.bias,   _np(sd[f"encoder.layer.{i}.output.dense.bias"]))
                assign_via_feed(sess, blk.output_norm.gamma,   _np(sd[f"encoder.layer.{i}.output.LayerNorm.weight"]))
                assign_via_feed(sess, blk.output_norm.beta,    _np(sd[f"encoder.layer.{i}.output.LayerNorm.bias"]))

                if i % 4 == 0 or i == len(tf_model.encoder_layers) - 1:
                    print(f"[encoder {i}] weights mapped")

            # pooler (검증용, dense_vecs에는 사용 안 함)
            if "pooler.dense.weight" in sd and "pooler.dense.bias" in sd:
                assign_via_feed(sess, tf_model.pooler.kernel, _np(sd["pooler.dense.weight"]).T)
                assign_via_feed(sess, tf_model.pooler.bias,   _np(sd["pooler.dense.bias"]))
                print("[pooler] initialized")

            # colbert
            if col_W is not None:
                assign_via_feed(sess, tf_model.colbert_linear.kernel, col_W.T)
                if hasattr(tf_model.colbert_linear, "bias") and tf_model.colbert_linear.bias is not None:
                    if col_b is None:
                        col_b = np.zeros((col_W.shape[0],), dtype=np.float32)
                    assign_via_feed(sess, tf_model.colbert_linear.bias, col_b)
                print(f"[colbert] initialized (out_dim={col_W.shape[0]}, in_dim={col_W.shape[1]})")
            else:
                print("[colbert] weights not found; skipping")

            # 내장 Sanity Check: L0가 PT와 붙는지 확인 (저장 이전)
            try:
                ids = tokenizer(["hello"], padding=True, truncation=True, max_length=8)
                inp_ids = np.array(ids["input_ids"], dtype=np.int32)
                att_msk = np.array(ids["attention_mask"], dtype=np.int32)
                pt_l0 = _pt_style_l0(sd, inp_ids, att_msk, padding_idx=1, eps=1e-5)
                tf_l0 = sess.run(hidd32[0], feed_dict={input_ids: inp_ids, attention_mask: att_msk})
                mse_l0 = np.mean((pt_l0 - tf_l0) ** 2)
                print(f"[sanity] L0 MSE vs PT: {mse_l0:.8f}")
                assert mse_l0 < 1e-6, "Embedding+LayerNorm (L0) mismatch; abort saving!"
            except Exception as e:
                raise

            # Signatures
            sig_default = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                outputs={"dense_vecs": dense32, "last_hidden_state": last32, "hidden_states": hidd32,
                         "colbert_vecs": col32},
            )
            sig_int64 = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                inputs={"input_ids_i64": input_ids_i64, "attention_mask_i64": attention_mask_i64},
                outputs={"dense_vecs_i64": dense64, "last_hidden_state_i64": last64, "hidden_states_i64": hidd64,
                         "colbert_vecs_i64": col64},
            )

            export_dir = os.path.join(out_root, "model")
            os.makedirs(export_dir, exist_ok=True)

            # ★ 여기서 '모든 변수'를 모아 커스텀 Saver를 만든다
            var_list = _collect_all_variables_for_saver(tf_model)
            saver = tf.compat.v1.train.Saver(
                var_list=var_list,
                write_version=tf.compat.v1.train.SaverDef.V2,
                save_relative_paths=True,
            )

            builder = tf.compat.v1.saved_model.Builder(export_dir)
            builder.add_meta_graph_and_variables(
                sess,
                tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={"serving_default": sig_default, "serving_int64": sig_int64},
                clear_devices=True,
                saver=saver,  # ★ 커스텀 Saver 지정 (중요)
            )
            builder.save()
            print(f"[export] TF1 SavedModel saved to: {export_dir}")

    tokenizer.save_pretrained(out_root)
    print(f"[export] tokenizer saved to: {out_root}")

def _as_tf_variable(v):
    """KerasVariable -> tf.Variable 로 변환 (그래프 모드). 이미 tf.Variable이면 그대로 반환."""
    try:
        # 가장 안전: 이미 tf.Variable 계열이면 그대로 사용
        if isinstance(v, tf.Variable):
            return v
    except Exception:
        pass
    # Keras 3의 래퍼가 내부 변수에 접근자를 제공할 수 있음
    inner = getattr(v, "variable", None)
    if isinstance(inner, tf.Variable):
        return inner
    inner2 = getattr(v, "_variable", None)
    if isinstance(inner2, tf.Variable):
        return inner2
    # 일부는 .value가 property일 수 있으나, method인 경우가 있어 호출/반환 지양
    if hasattr(v, "value") and not callable(getattr(v, "value")):
        inner3 = getattr(v, "value")
        if isinstance(inner3, tf.Variable):
            return inner3
    return v

def _collect_all_variables_for_saver(tf_model):
    """
    Saver에 전달할 '완전한' 변수 목록을 구성.
    - Keras 3의 tf_model.variables (KerasVariable) 포함
    - TF1 컬렉션의 global/trainable/model 변수 포함
    - 이름으로 dedup
    """
    vars_from_keras = [_as_tf_variable(v) for v in getattr(tf_model, "variables", [])]

    vars_global   = list(tf.compat.v1.global_variables())
    vars_train    = list(tf.compat.v1.trainable_variables())
    try:
        vars_model = list(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.MODEL_VARIABLES))
    except Exception:
        vars_model = []

    all_vars = vars_from_keras + vars_global + vars_train + vars_model

    # 이름 기준 dedup (handle/참조가 달라도 같은 변수면 이름이 동일)
    dedup = []
    seen = set()
    for v in all_vars:
        try:
            name = v.name  # e.g. 'bge-m3-tensorflow/...:0'
        except Exception:
            continue
        if name not in seen:
            seen.add(name)
            dedup.append(v)

    # 디버그: 저장할 변수 개수/샘플 이름 출력
    print(f"[saver] variables to save: {len(dedup)}")
    for nm in list(seen)[:5]:
        print("  -", nm)

    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--out",   type=str, default="./converted_bge_m3_tf1_java_fixed")
    args = parser.parse_args()
    export_tf1_saved_model(args.model, args.out)

if __name__ == "__main__":
    main()
