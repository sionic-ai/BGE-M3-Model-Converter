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
    inputs_tf_w_padding_attnFixed = inputs_tf_w_padding.copy()
    inputs_tf_w_padding_attnFixed['attention_mask'] = tf.where(inputs_tf_w_padding['attention_mask'] == 0, -9999999, 0)
    tf_model = load_tf_model(model_path_tf).signatures["serving_default"]

    loguru.logger.info("Tensorflow] Model output".ljust(50, "-"))
    with tf.device("/GPU:0"):
        output_tf = tf_model(**inputs_tf)
        output_tf_w_padding = tf_model(**inputs_tf_w_padding)
        output_tf_w_padding_attnFixed = tf_model(**inputs_tf_w_padding_attnFixed)
        loguru.logger.info("output without padding (GT)".ljust(50, "-"))
        loguru.logger.info(output_tf['hidden_states'][-1][:,0])
        loguru.logger.info("="*50)
        loguru.logger.info("output with padding".ljust(50, "-"))
        loguru.logger.info(output_tf_w_padding['hidden_states'][-1][:,0])
        loguru.logger.info("="*50)
        loguru.logger.info("output with padding (attention fixed)".ljust(50, "-"))
        loguru.logger.info(output_tf_w_padding_attnFixed['hidden_states'][-1][:,0])
        loguru.logger.info("="*50)
        err_tf = tf.abs(output_tf['hidden_states'][-1][:,0] - output_tf_w_padding['hidden_states'][-1][:,0])
        loguru.logger.info("Error".ljust(50, "-"))
        loguru.logger.info(tf.reduce_mean(err_tf))
        loguru.logger.info("="*50)
        err_tf_attnFixed = tf.abs(output_tf_w_padding['hidden_states'][-1][:,0] - output_tf_w_padding_attnFixed['hidden_states'][-1][:,0])
        loguru.logger.info("Error (attention fixed)".ljust(50, "-"))
        loguru.logger.info(tf.reduce_mean(err_tf_attnFixed))
        loguru.logger.info("="*50)
        


if __name__ == "__main__":
    main()