import os
import subprocess
from typing import Tuple

import torch
from transformers import AutoModel, AutoTokenizer


class _LastHiddenStateWrapper(torch.nn.Module):
    """Wrap HF model to export only last_hidden_state to ONNX."""

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        # HF BaseModelOutput has attribute last_hidden_state
        return outputs.last_hidden_state


def _export_onnx(model_name: str, onnx_path: str, max_len: int = 128, opset: int = 15) -> Tuple[str, str]:
    """Export HF PyTorch model to ONNX with only last_hidden_state output.

    Returns (onnx_path, external_data_dir)
    """
    device = torch.device("cpu")
    hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    hf_model.eval()

    wrapper = _LastHiddenStateWrapper(hf_model).to(device)
    wrapper.eval()

    # Dummy inputs
    bsz = 2
    dummy_ids = torch.ones((bsz, max_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((bsz, max_len), dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Use external data to avoid 2GB protobuf limit
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
        use_external_data_format=True,
    )

    return onnx_path, os.path.dirname(onnx_path)


def _convert_onnx_to_tf_saved_model(onnx_path: str, saved_model_dir: str):
    """Convert ONNX to TF SavedModel using onnx-tf CLI to mimic old-style export.

    Requires `onnx-tf` to be installed and available on PATH.
    """
    os.makedirs(saved_model_dir, exist_ok=True)
    cmd = [
        "onnx-tf",
        "convert",
        "-t",
        "tf",
        "-o",
        saved_model_dir,
        "-i",
        onnx_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "onnx-tf CLI not found. Please install with: pip install onnx onnx-tf"
        )


def export_old_style(model_name: str, out_dir: str, opset: int = 15, max_len: int = 128):
    """Export a SavedModel that follows the old style (no variables, ONNX-TF conversion).

    - Output structure:
        out_dir/
          model/  (SavedModel from onnx-tf)
          tokenizer.json, tokenizer_config.json, special_tokens_map.json
    - Signature:
        inputs: input_ids:int64, attention_mask:int64
        outputs: last_hidden_state:float32 [B, T, H]
    - Notes:
        - This path does NOT include colbert_vecs nor hidden_states.
        - Requires: pip install onnx onnx-tf
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Export ONNX with only last_hidden_state
    onnx_path = os.path.join(out_dir, "model.onnx")
    _export_onnx(model_name, onnx_path, max_len=max_len, opset=opset)

    # 2) Convert ONNX -> TF SavedModel (old-style: constants, no ResourceVariables)
    saved_model_dir = os.path.join(out_dir, "model")
    _convert_onnx_to_tf_saved_model(onnx_path, saved_model_dir)

    # 3) Save tokenizer alongside (root of out_dir)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.save_pretrained(out_dir)

    print(f"Old-style export completed: {out_dir}")


if __name__ == "__main__":
    # Example: python old_style_export.py
    model = "BAAI/bge-m3"
    out = "./new_model_oldstyle"
    export_old_style(model, out)
