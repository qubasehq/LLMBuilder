# Model Export

Model export allows you to convert your trained LLMBuilder models into different formats for deployment.

## Export Overview

LLMBuilder supports multiple export formats:
- GGUF Format - For CPU inference with llama.cpp
- ONNX Format - For cross-platform deployment
- Quantized Models - For smaller, faster models

## Quick Start

### CLI Export

```bash
# Export to GGUF format
llmbuilder export gguf \
  ./model/model.pt \
  --output model.gguf \
  --quantization q4_0

# Export to ONNX format
llmbuilder export onnx \
  ./model/model.pt \
  --output model.onnx \
  --opset 11

# Quantize model
llmbuilder export quantize \
  ./model/model.pt \
  --output quantized_model.pt \
  --method dynamic \
  --bits 8
```

### Python API Export

```python
from llmbuilder.export import export_gguf, export_onnx, quantize_model

# Export to GGUF
export_gguf(
    model_path="./model/model.pt",
    output_path="model.gguf",
    quantization="q4_0"
)

# Export to ONNX
export_onnx(
    model_path="./model/model.pt",
    output_path="model.onnx",
    opset_version=11
)

# Quantize model
quantize_model(
    model_path="./model/model.pt",
    output_path="quantized_model.pt",
    method="dynamic",
    bits=8
)
```

## Export Formats

### 1. GGUF Format

GGUF is optimized for CPU inference with llama.cpp:

```python
from llmbuilder.export import GGUFExporter

exporter = GGUFExporter(
    model_path="./model/model.pt",
    tokenizer_path="./tokenizer"
)

# Export with different quantization levels
exporter.export(
    output_path="model_f16.gguf",
    quantization="f16"          # Full precision
)

exporter.export(
    output_path="model_q8.gguf",
    quantization="q8_0"         # 8-bit quantization
)

exporter.export(
    output_path="model_q4.gguf",
    quantization="q4_0"         # 4-bit quantization
)
```

**Quantization Options:**
- `f16`: 16-bit floating point (best quality, larger size)
- `q8_0`: 8-bit quantization (good quality, medium size)
- `q4_0`: 4-bit quantization (lower quality, smallest size)

### 2. ONNX Format

ONNX for cross-platform deployment:

```python
from llmbuilder.export import ONNXExporter

exporter = ONNXExporter(
    model_path="./model/model.pt",
    tokenizer_path="./tokenizer"
)

# Export for different targets
exporter.export(
    output_path="model_cpu.onnx",
    target="cpu",
    opset_version=11
)
```

## Next Steps

- **[Deployment Guide]** - Deploy your models
- **[Fine-tuning Guide](fine-tuning.md)** - Improve your models
- **[Generation Guide](generation.md)** - Generate text with exported models

<div align="center">
  <p>Start with GGUF export for CPU deployment.</p>
</div>