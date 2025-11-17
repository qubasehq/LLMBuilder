# Frequently Asked Questions

Common questions and answers about LLMBuilder usage, troubleshooting, and best practices.

## 🚀 Getting Started

### Q: What are the minimum system requirements?

**A:** LLMBuilder requires:

- **Python 3.8+** (3.9+ recommended)
- **4GB RAM** minimum (8GB+ recommended)
- **2GB free disk space** for installation and basic models
- **Optional:** NVIDIA GPU with 4GB+ VRAM for faster training

### Q: Should I use CPU or GPU for training?

**A:**

- **CPU**: Good for learning, small models, and development. Use `preset="cpu_small"`
- **GPU**: Recommended for production training and larger models. Use `preset="gpu_medium"` or `preset="gpu_large"`
- **Mixed**: Start with CPU for prototyping, then move to GPU for final training

### Q: How long does it take to train a model?

**A:** Training time depends on several factors:

- **Small model (10M params)**: 30 minutes - 2 hours on CPU, 5-15 minutes on GPU
- **Medium model (50M params)**: 2-8 hours on CPU, 30 minutes - 2 hours on GPU
- **Large model (200M+ params)**: Days on CPU, 2-12 hours on GPU

## 🔧 Configuration

### Q: Which configuration preset should I use?

**A:** Choose based on your hardware and use case:

| Preset | Use Case | Hardware | Model Size | Training Time |
|--------|----------|----------|------------|---------------|
| `tiny` | Testing, debugging | Any | ~1M params | Minutes |
| `cpu_small` | Learning, development | CPU | ~10M params | Hours |
| `gpu_medium` | Production training | Single GPU | ~50M params | Hours |
| `gpu_large` | High-quality models | High-end GPU | ~200M+ params | Days |

### Q: How do I customize model architecture?

**A:** Modify the model configuration:

```python
from llmbuilder.config import ModelConfig

config = ModelConfig(
    vocab_size=16000,      # Match your tokenizer
    num_layers=12,         # More layers = more capacity
    num_heads=12,          # Should divide embedding_dim evenly
    embedding_dim=768,     # Larger = more capacity
    max_seq_length=1024,   # Longer sequences = more memory
    dropout=0.1            # Higher = more regularization
)
```

### Q: What vocabulary size should I use?

**A:** Vocabulary size depends on your data and use case:

- **8K-16K**: Small datasets, specific domains
- **16K-32K**: General purpose, balanced size
- **32K-64K**: Large datasets, multilingual models
- **64K+**: Very large datasets, maximum coverage

## 📊 Data and Training

### Q: How much training data do I need?

**A:** Data requirements vary by model size and quality goals:

- **Minimum**: 1MB of text (~200K words) for basic functionality
- **Recommended**: 10MB+ of text (~2M words) for good quality
- **Optimal**: 100MB+ of text (~20M words) for high quality
- **Production**: 1GB+ of text (~200M words) for best results

### Q: What file formats are supported for training data?

**A:** LLMBuilder supports:

- **Text files**: `.txt`, `.md` (best quality)
- **Documents**: `.pdf`, `.docx` (good quality)
- **Web content**: `.html`, `.htm` (moderate quality)
- **Presentations**: `.pptx` (basic support)
- **Data files**: `.csv`, `.json` (with proper formatting)

### Q: How do I handle out-of-memory errors?

**A:** Try these solutions in order:

1. **Reduce batch size**:

```python
config.training.batch_size = 4  # or even 2
```

2. **Enable gradient checkpointing**:

```python
config.model.gradient_checkpointing = True
```

3. **Use gradient accumulation**:

```python
config.training.gradient_accumulation_steps = 4
```

4. **Reduce sequence length**:

```python
config.model.max_seq_length = 512
```

5. **Use CPU training**:

```python
config.system.device = "cpu"
```

### Q: My model isn't learning (loss not decreasing). What's wrong?

**A:** Common causes and solutions:

1. **Learning rate too high**: Reduce to 1e-4 or 1e-5
2. **Learning rate too low**: Increase to 3e-4 or 5e-4
3. **Bad data**: Check for corrupted or repetitive text
4. **Wrong tokenizer**: Ensure vocab_size matches tokenizer
5. **Insufficient warmup**: Increase warmup_steps to 1000+

### Q: How do I know if my model is overfitting?

**A:** Signs of overfitting:

- Training loss decreases but validation loss increases
- Generated text is repetitive or memorized
- Model performs poorly on new data

**Solutions:**

- Increase dropout rate (0.1 → 0.2)
- Add weight decay (0.01)
- Use early stopping
- Get more training data
- Reduce model size

## 🎯 Text Generation

### Q: How do I improve generation quality?

**A:** Try these techniques:

1. **Adjust temperature**:
   - Lower (0.3-0.7): More focused, predictable
   - Higher (0.8-1.2): More creative, diverse

2. **Use nucleus sampling**:

```python
config = GenerationConfig(
    temperature=0.8,
    top_p=0.9,      # Nucleus sampling
    top_k=50        # Top-k sampling
)
```

3. **Add repetition penalty**:

```python
config.repetition_penalty = 1.1
```

4. **Better prompts**:
   - Be specific and clear
   - Provide context and examples
   - Use consistent formatting

### Q: Why is my generated text repetitive?

**A:** Common causes and fixes:

1. **Insufficient training**: Train for more epochs
2. **Poor sampling**: Use top-p/top-k sampling instead of greedy
3. **Low temperature**: Increase temperature to 0.8+
4. **Add repetition penalty**: Set to 1.1-1.3
5. **Prevent n-gram repetition**: Set `no_repeat_ngram_size=3`

### Q: How do I make generation faster?

**A:** Speed optimization techniques:

1. **Use GPU**: Much faster than CPU
2. **Reduce max_tokens**: Generate shorter responses
3. **Use greedy decoding**: Set `do_sample=False`
4. **Enable model compilation**: Set `compile=True` (PyTorch 2.0+)
5. **Quantize model**: Use 8-bit or 16-bit precision

## 🔄 Fine-tuning

### Q: When should I fine-tune vs. train from scratch?

**A:**

- **Fine-tune when**: You have a pre-trained model and domain-specific data
- **Train from scratch when**: You have lots of data and need full control
- **Fine-tuning advantages**: Faster, less data needed, preserves general knowledge
- **Training advantages**: Full customization, no dependency on base model

### Q: What's the difference between LoRA and full fine-tuning?

**A:**

| Aspect | LoRA | Full Fine-tuning |
|--------|------|------------------|
| **Memory** | Low (~1% of params) | High (all params) |
| **Speed** | Fast | Slower |
| **Quality** | Good for most tasks | Best possible |
| **Flexibility** | Limited adaptation | Full adaptation |
| **Use case** | Domain adaptation | Major architecture changes |

### Q: How do I prevent catastrophic forgetting during fine-tuning?

**A:** Use these techniques:

1. **Lower learning rate**: 1e-5 to 5e-5
2. **Fewer epochs**: 3-5 epochs usually sufficient
3. **Regularization**: Add weight decay (0.01)
4. **LoRA**: Preserves base model weights
5. **Mixed training**: Include general data with domain data

## 🚀 Deployment

### Q: How do I deploy my trained model?

**A:** LLMBuilder supports multiple deployment options:

1. **GGUF format** (for llama.cpp):

```bash
llmbuilder export gguf model.pt --output model.gguf --quantization q4_0
```

2. **ONNX format** (for cross-platform):

```bash
llmbuilder export onnx model.pt --output model.onnx
```

3. **Quantized PyTorch** (for production):

```bash
llmbuilder export quantize model.pt --output model_int8.pt --bits 8
```

### Q: Which export format should I choose?

**A:** Choose based on your deployment target:

- **GGUF**: CPU inference, llama.cpp compatibility, edge devices
- **ONNX**: Cross-platform, mobile apps, cloud services
- **Quantized PyTorch**: PyTorch ecosystem, balanced performance
- **HuggingFace**: Easy sharing, transformers compatibility

### Q: How do I reduce model size for deployment?

**A:** Size reduction techniques:

1. **Quantization**: 8-bit (50% smaller) or 4-bit (75% smaller)
2. **Pruning**: Remove least important weights
3. **Distillation**: Train smaller model to mimic larger one
4. **Architecture optimization**: Use efficient attention mechanisms

## 🐛 Troubleshooting

### Q: I get "CUDA out of memory" errors. What should I do?

**A:** Try these solutions:

1. **Reduce batch size**: Start with batch_size=1
2. **Enable gradient checkpointing**: Trades compute for memory
3. **Use gradient accumulation**: Simulate larger batches
4. **Reduce sequence length**: Shorter sequences use less memory
5. **Use CPU**: Slower but no memory limits
6. **Clear GPU cache**: `torch.cuda.empty_cache()`

### Q: Training is very slow. How can I speed it up?

**A:** Speed optimization:

1. **Use GPU**: 10-100x faster than CPU
2. **Increase batch size**: Better GPU utilization
3. **Enable mixed precision**: `fp16` or `bf16`
4. **Use multiple GPUs**: Distributed training
5. **Optimize data loading**: More workers, pin memory
6. **Compile model**: PyTorch 2.0 compilation

### Q: My tokenizer produces weird results. What's wrong?

**A:** Common tokenizer issues:

1. **Wrong vocabulary size**: Must match model config
2. **Insufficient training data**: Need diverse text corpus
3. **Character coverage too low**: Increase to 0.9999
4. **Wrong model type**: BPE usually works best
5. **Missing special tokens**: Include `<pad>`, `<unk>`, etc.

### Q: Generated text contains strange characters or formatting

**A:** Text cleaning solutions:

1. **Improve data cleaning**: Remove unwanted characters
2. **Filter by language**: Keep only desired languages
3. **Normalize text**: Fix encoding issues
4. **Add text filters**: Remove specific patterns
5. **Better tokenizer training**: Use cleaner training data

## 💡 Best Practices

### Q: What are the most important best practices?

**A:** Key recommendations:

1. **Start small**: Begin with tiny models and scale up
2. **Clean your data**: Quality over quantity
3. **Monitor training**: Watch loss curves and generation quality
4. **Save checkpoints**: Protect against failures
5. **Validate everything**: Test configurations before long training
6. **Document experiments**: Keep track of what works

### Q: How do I choose hyperparameters?

**A:** Hyperparameter selection guide:

1. **Learning rate**: Start with 3e-4, adjust based on loss curves
2. **Batch size**: Largest that fits in memory
3. **Model size**: Balance quality needs with resources
4. **Sequence length**: Match your use case requirements
5. **Dropout**: 0.1 is usually good, increase if overfitting

### Q: How do I evaluate model quality?

**A:** Evaluation methods:

1. **Perplexity**: Lower is better (< 20 is good)
2. **Generation quality**: Manual inspection of outputs
3. **Task-specific metrics**: BLEU, ROUGE for specific tasks
4. **Human evaluation**: Best but most expensive
5. **Automated metrics**: Coherence, fluency scores

## 🆘 Getting Help

### Q: Where can I get help if I'm stuck?

**A:** Support resources:

1. **Documentation**: Complete guides and examples
2. **GitHub Issues**: Report bugs and request features
3. **GitHub Discussions**: Community Q&A
4. **Examples**: Working code samples
5. **Stack Overflow**: Tag questions with `llmbuilder`

### Q: How do I report a bug?

**A:** When reporting bugs, include:

1. **LLMBuilder version**: `llmbuilder --version`
2. **Python version**: `python --version`
3. **Operating system**: Windows/macOS/Linux
4. **Hardware**: CPU/GPU specifications
5. **Error message**: Full traceback
6. **Minimal example**: Code to reproduce the issue
7. **Configuration**: Model and training configs used

### Q: How can I contribute to LLMBuilder?

**A:** Ways to contribute:

1. **Report bugs**: Help improve stability
2. **Request features**: Suggest improvements
3. **Submit PRs**: Code contributions welcome
4. **Improve docs**: Fix typos, add examples
5. **Share examples**: Help other users
6. **Test releases**: Try beta versions

---

!!! tip "Still have questions?"
    If you can't find the answer here, check our [GitHub Discussions](https://github.com/Qubasehq/llmbuilder/discussions) or create a new issue. The community is always happy to help!
