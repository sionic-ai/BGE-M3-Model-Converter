# torch_tf_validator.py
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

TAG_SERVE = tf.saved_model.SERVING
SIG_NAME = "serving_default"

def _tensor_shape_to_list(tensor_shape_proto):
    dims = tensor_shape_proto.dim
    if not dims:
        return None
    return [d.size for d in dims]

def inspect_signature(model_dir: str):
    print(f"\n[1] SavedModel 로드 및 시그니처 점검: {model_dir}")
    g = tf.Graph()
    with g.as_default():
        with tf.compat.v1.Session(graph=g) as sess:
            meta_graph_def = tf.compat.v1.saved_model.load(sess, [TAG_SERVE], model_dir)
            sigs = meta_graph_def.signature_def
            if SIG_NAME not in sigs:
                raise RuntimeError(f"Signature '{SIG_NAME}' not found. Available: {list(sigs.keys())}")
            sig = sigs[SIG_NAME]
            print(f" - 사용 시그니처: {SIG_NAME}")
            print(" - 입력들:")
            for k, tinfo in sig.inputs.items():
                print(f"    • key='{k}', dtype={tf.DType(tinfo.dtype).name}, "
                      f"shape={_tensor_shape_to_list(tinfo.tensor_shape)}, name='{tinfo.name}'")
            print(" - 출력들:")
            for k, tinfo in sig.outputs.items():
                print(f"    • key='{k}', dtype={tf.DType(tinfo.dtype).name}, "
                      f"shape={_tensor_shape_to_list(tinfo.tensor_shape)}, name='{tinfo.name}'")

def run_session_once(model_dir: str, dtype=np.int32, batch=2, seqlen=12):
    print(f"\n[2] TF1 세션으로 1회 추론 실행 (입력 dtype={np.dtype(dtype).name}, B={batch}, T={seqlen})")
    g = tf.Graph()
    with g.as_default():
        with tf.compat.v1.Session(graph=g) as sess:
            meta_graph_def = tf.compat.v1.saved_model.load(sess, [TAG_SERVE], model_dir)
            sig = meta_graph_def.signature_def[SIG_NAME]

            t_input_ids = g.get_tensor_by_name(sig.inputs["input_ids"].name)
            t_attention = g.get_tensor_by_name(sig.inputs["attention_mask"].name)
            t_last = g.get_tensor_by_name(sig.outputs["last_hidden_state"].name)
            t_colbert = g.get_tensor_by_name(sig.outputs["colbert_vecs"].name)

            input_ids = np.random.randint(10, 1000, size=(batch, seqlen)).astype(dtype)
            attention_mask = np.ones((batch, seqlen), dtype=dtype)

            last_h, colbert = sess.run(
                [t_last, t_colbert],
                feed_dict={t_input_ids: input_ids, t_attention: attention_mask}
            )
            print(" - last_hidden_state:", last_h.shape, last_h.dtype)
            print(" - colbert_vecs     :", colbert.shape, colbert.dtype)
            assert last_h.dtype == np.float32 and colbert.dtype == np.float32
            assert last_h.shape == (batch, seqlen, 1024)
            print(colbert.shape)
            #assert colbert.shape == (batch, seqlen, 1024)
            print(" ✔ 세션 추론 성공")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="converted_bge_m3_tf1_java_fixed/model", help="Path to SavedModel directory (…/model)")
    args = ap.parse_args()

    inspect_signature(args.model_dir)
    run_session_once(args.model_dir, dtype=np.int32, batch=2, seqlen=12)

    print("\n[3] (의도적) int64로 재실행 → 보통 dtype mismatch로 실패하거나 내부 캐스팅 없으면 에러")
    try:
        run_session_once(args.model_dir, dtype=np.int64, batch=2, seqlen=12)
        print("※ int64 입력이 통과하면, 시그니처가 int64이거나 내부 캐스팅이 있는 경우입니다.")
    except Exception as e:
        print(" ✔ 기대된 실패(입력 dtype 불일치):", type(e).__name__, str(e)[:200], "…")

if __name__ == "__main__":
    main()
