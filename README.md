# BGE-M3 Model Converter

A tool for manual conversion of BGE-M3 models with preserved trainable variables and direct control over model outputs.

## Features

- Manual conversion of BGE-M3 models with preserved model architecture
- Direct control over model outputs (colbert, sparse, etc.)
- Resolution for ONNX conversion failures
- Solution for protobuf 2GB file size limitation
- Preservation of trainable variable information in the model structure

## Project Structure

```
├── BGEM3TFModel.py           # Model architecture definition
├── BGEM3WeightConverter.py   # Weight conversion and saving implementation
└── model_conversion_validator.py  # Conversion validation code
```

## Prerequisites

- Python 3.11
- Virtual environment
- Git LFS
- transformers==4.47.1
- keras==3.7.0
- tensorflow==2.16.2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sionic-ai/convert-keras-bge-m3
cd convert-keras-bge-m3
```

2. Install Git LFS and clone the BGE-M3 model:
```bash
git lfs install 
git clone https://huggingface.co/BAAI/bge-m3
```

3. Set up Python virtual environment:
```bash
uv venv
source ./venv/bin/activate
uv sync
# uv add -r requirements.txt
```

## Usage

1. Convert the model weights:
```bash
uv run BGEM3WeightConverter.py
```

2. Validate the converted model:
```bash
uv run model_conversion_validator.py
```

## Model Architecture

The converter works with XLMRobertaModel architecture with the following specifications:

- Hidden size: 1024
- Attention heads: 16
- Hidden layers: 24
- Maximum position embeddings: 8194
- Intermediate size: 4096
- Vocabulary size: 250002

## Input/Output Example

```

Encoder Layer 24:
PyTorch shape: torch.Size([2, 11, 1024])
    dims: [batch_size=2, seq_len=11, hidden_dim=1024]
TensorFlow shape: (2, 11, 1024)
    dims: [batch_size=2, seq_len=11, hidden_dim=1024]
  -> MSE: 0.000000
  -> CLS Token Cosine Similarity: 1.000000
[TensorFlow] Final Embeddings Shape: (2, 1024)

=== 3) PT vs. TF 최종 임베딩 비교 ===
[[-2.0583858  -0.19201966 -1.1751263  ...  0.71480525  0.02913969
   0.08781114]
 [-2.0583858  -0.19201966 -1.1751263  ...  0.71480525  0.02913969
   0.08781114]]
[[-2.058385   -0.19201875 -1.175129   ...  0.7148077   0.02913864
   0.08781095]
 [-2.058385   -0.19201948 -1.1751292  ...  0.71480644  0.0291374
   0.08781061]]
===== Queries =====
[0] 이 모델은 무엇을 하는 모델인가요?
[1] 이 모델은 무엇을 하는 모델인가요?

===== PyTorch Embeddings (shape) =====
(2, 1024)
===== TF Embeddings (shape) =====
(2, 1024)

#Java
# text : "안녕하세요"
# input_ids : 0, 107687, 2
#-0.63850623, 0.5008155, -0.8110449, 0.1603587, -0.31579554, 

# Python
# input_text: 안녕하세요
# embedding_dimension : torch.Size([1, 1024])
# tensor([
# -0.6385,  0.5008, -0.8110,  0.1604, -0.3158,
# [-0.6385, 0.5008, -0.8110, 0.1604, -0.3158, ...]
```

## Contributing

Feel free to open issues and pull requests for any improvements or bug fixes.

## License

MIT License

```
Copyright (c) 2024 [Sionia AI Inc.]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

Note: This project is specifically designed for BGE-M3 model conversion and may require specific hardware configurations and dependencies. Please ensure your environment meets the requirements before proceeding with the conversion.
