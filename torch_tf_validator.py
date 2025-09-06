import loguru 

from transformers import AutoTokenizer, AutoModel
import tensorflow as tf 
import torch 

def load_torch_model(model_path):
    model = AutoModel.from_pretrained(model_path)
    return model


def load_tf_model(model_path):
    with tf.device("/CPU:0"):
        model = tf.saved_model.load(model_path)
    return model


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def tokenize_wo_padding(tokenizer, text, return_tensors="pt"):
    return tokenizer(text, padding=False, return_tensors=return_tensors)


def tokenize_w_padding(tokenizer, text, return_tensors="pt", max_length=512):
    return tokenizer(text, padding="max_length", max_length=max_length, return_tensors=return_tensors)


def call_signature(sig, input_ids, attention_mask, token_type_ids=None):
    """Adapt arguments to a SavedModel signature: keys and dtypes."""
    spec_kwargs = sig.structured_input_signature[1]

    def prepare(name, value):
        if name not in spec_kwargs:
            return None
        if value is None and name == "token_type_ids":
            value = tf.zeros_like(input_ids)
        want = spec_kwargs[name].dtype
        if hasattr(value, "dtype") and value.dtype != want:
            value = tf.cast(value, want)
        return value

    kwargs = {}
    x = prepare("input_ids", input_ids)
    if x is not None:
        kwargs["input_ids"] = x
    x = prepare("attention_mask", attention_mask)
    if x is not None:
        kwargs["attention_mask"] = x
    x = prepare("token_type_ids", token_type_ids)
    if x is not None:
        kwargs["token_type_ids"] = x
    return sig(**kwargs)


def main():
    # Load the model
    model_path = "BAAI/bge-m3"
    model_path_tf = "/workspace/BGE-M3-Model-Converter/model"
    model = load_torch_model(model_path)
    tokenizer = load_tokenizer(model_path)

    # Tokenize the text
    text = "Hello, my dog is cute"
    inputs = tokenize_wo_padding(tokenizer, text)
    inputs_w_padding = tokenize_w_padding(tokenizer, text)

    # Get the output from the model
    loguru.logger.info("Torch] Model output".ljust(50, "-"))
    model.eval().to("cuda")
    with torch.no_grad():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inputs_w_padding = {k: v.to("cuda") for k, v in inputs_w_padding.items()}

        output = model(**inputs)
        output_w_padding = model(**inputs_w_padding)
        loguru.logger.info("output without padding (GT)".ljust(50, "-"))
        loguru.logger.info(output['last_hidden_state'][:, 0])
        loguru.logger.info("="*50)
        loguru.logger.info("output with padding".ljust(50, "-"))
        loguru.logger.info(output_w_padding['last_hidden_state'][:, 0])
        loguru.logger.info("="*50)
        err = torch.abs(output['last_hidden_state'][:, 0] - output_w_padding['last_hidden_state'][:, 0])
        loguru.logger.info("Error".ljust(50, "-"))
        loguru.logger.info(err.mean())
    
    inputs_tf = tokenize_wo_padding(tokenizer, text, return_tensors="tf")
    inputs_tf_w_padding = tokenize_w_padding(tokenizer, text, return_tensors="tf")
    loaded = load_tf_model(model_path_tf)
    # Use the default 2-input signature
    sigs = loaded.signatures
    tf_model = sigs.get("serving_default")
    if tf_model is None:
        raise RuntimeError("serving_default signature not found")

    loguru.logger.info("Tensorflow] Model output".ljust(50, "-"))
    with tf.device("/GPU:0"):
        output_tf = call_signature(
            tf_model, inputs_tf["input_ids"], inputs_tf["attention_mask"], None
        )
        output_tf_w_padding = call_signature(
            tf_model,
            inputs_tf_w_padding["input_ids"],
            inputs_tf_w_padding["attention_mask"],
            None,
        )
        loguru.logger.info("output without padding (GT)".ljust(50, "-"))
        hs = output_tf['last_hidden_state']
        val_no_pad = hs[:, 0]
        loguru.logger.info(val_no_pad)
        loguru.logger.info("="*50)
        loguru.logger.info("output with padding".ljust(50, "-"))
        hsw = output_tf_w_padding['last_hidden_state']
        val_pad = hsw[:, 0]
        loguru.logger.info(val_pad)
        loguru.logger.info("="*50)
        err_tf = tf.abs(val_no_pad - val_pad)
        loguru.logger.info("Error".ljust(50, "-"))
        loguru.logger.info(tf.reduce_mean(err_tf))
        loguru.logger.info("="*50)
        


if __name__ == "__main__":
    main()
