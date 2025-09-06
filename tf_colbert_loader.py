import os
import argparse
import tensorflow as tf
from transformers import AutoTokenizer


def resolve_model_path(root_dir: str) -> str:
    """Return a path that contains saved_model.pb (root or <root>/model)."""
    cand1 = root_dir
    cand2 = os.path.join(root_dir, "model")
    if os.path.exists(os.path.join(cand2, "saved_model.pb")):
        return cand2
    if os.path.exists(os.path.join(cand1, "saved_model.pb")):
        return cand1
    raise FileNotFoundError(f"No SavedModel found under '{root_dir}' (checked '{cand2}' and '{cand1}')")


def call_signature(sig, input_ids, attention_mask, token_type_ids=None):
    """Call signature with best-effort arg set and dtype handling.
    Tries (int64,int32) x (3-key,2-key) in order.
    """
    last_err = None
    for dtype in (tf.int64, tf.int32):
        ii = tf.cast(input_ids, dtype)
        am = tf.cast(attention_mask, dtype)
        # Try 3-key first
        if token_type_ids is not None:
            try:
                tt = tf.cast(token_type_ids, dtype)
                return sig(input_ids=ii, attention_mask=am, token_type_ids=tt)
            except Exception as e:
                last_err = e
        # Then 2-key
        try:
            return sig(input_ids=ii, attention_mask=am)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to call signature with any supported argument pattern")


def main():
    parser = argparse.ArgumentParser(description="Load TF ColBERT SavedModel and print output shapes.")
    parser.add_argument("--model_dir", default="./converted_bge_m3", help="Path to SavedModel root (contains tokenizer files).")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenization max length.")
    parser.add_argument("--texts", nargs="*", default=[
        "이 모델은 무엇을 하는 모델인가요?",
        "bge-m3 tensorflow colbert vectors test",
    ])
    args = parser.parse_args()

    # Load tokenizer from the same root dir
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    inputs = tok(
        args.texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="tf",
    )
    # token_type_ids may be missing for XLM-R; create zeros if absent
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = tf.zeros_like(inputs["input_ids"], dtype=inputs["input_ids"].dtype)

    # Load SavedModel signature
    model_path = resolve_model_path(args.model_dir)
    loaded = tf.saved_model.load(model_path)
    sig = loaded.signatures.get("serving_default")
    if sig is None:
        raise RuntimeError("serving_default signature not found")

    # Call signature robustly
    outs = None
    try:
        outs = call_signature(sig, inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids"))
    except TypeError:
        # Try without token_type_ids
        outs = call_signature(sig, inputs["input_ids"], inputs["attention_mask"], None)

    # Print keys and shapes
    print("Signature outputs:")
    for k, v in outs.items():
        try:
            print(f"- {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        except Exception:
            print(f"- {k}: <non-tensor>")

    # Convenience checks
    if "last_hidden_state" in outs:
        print("last_hidden_state OK ->", tuple(outs["last_hidden_state"].shape))
    if "colbert_vecs" in outs:
        print("colbert_vecs OK ->", tuple(outs["colbert_vecs"].shape))
    else:
        print("colbert_vecs not present in signature outputs.")
    # hidden_states is no longer returned in serving by design


if __name__ == "__main__":
    main()
