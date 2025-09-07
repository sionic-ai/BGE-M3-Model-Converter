# tf1_session_validator.py
import argparse
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.python.saved_model import loader, tag_constants

SIG = "serving_default"

def inspect_and_run(model_dir: str, b=2, t=12):
    print(f"[inspect] {model_dir}")
    g = tf.Graph()
    with g.as_default():
        with tf.compat.v1.Session(graph=g) as sess:
            meta = loader.load(sess, [tag_constants.SERVING], model_dir)
            sig = meta.signature_def[SIG]

            t_ids  = g.get_tensor_by_name(sig.inputs["input_ids"].name)
            t_msk  = g.get_tensor_by_name(sig.inputs["attention_mask"].name)
            t_last = g.get_tensor_by_name(sig.outputs["last_hidden_state"].name)
            t_colb = g.get_tensor_by_name(sig.outputs["colbert_vecs"].name)

            print(" - inputs :", sig.inputs)
            print(" - outputs:", sig.outputs)

            ids = np.random.randint(10, 1000, size=(b, t)).astype(np.int32)
            msk = np.ones((b, t), dtype=np.int32)

            last, colb = sess.run([t_last, t_colb], feed_dict={t_ids: ids, t_msk: msk})
            print("last_hidden_state:", last.shape, last.dtype)
            print("colbert_vecs     :", colb.shape, colb.dtype)
            print("✔ TF1 Session run OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=False, default="./converted_bge_m3_tf1_v1")
    args = ap.parse_args()
    inspect_and_run(args.dir)
