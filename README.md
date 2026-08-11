# Transformer Runtime & Inference Simulation

A lightweight **Transformer inference runtime simulation built with Python and NumPy**. This project models key components of a modern Transformer inference stack, including multi-head self-attention, FlashAttention-style block processing, GEMM-based attention, FP32/FP16/INT8 precision simulation, KV-cache management, memory tracking, performance profiling, and autoregressive token generation.

> **Note:** This project is an inference/runtime simulation rather than a trained language model. The model weights are randomly initialized and the quantization implementation simulates reduced-precision numerical behavior rather than performing real hardware bit-packing.

## Features

* Transformer-style model architecture
* Multi-head self-attention
* Two attention execution paths:

  * GEMM-based attention
  * FlashAttention-style block-wise attention
* FP32, FP16, and simulated INT8 precision
* KV-cache simulation
* Memory allocation and memory-usage tracking
* Runtime performance profiling
* Autoregressive token generation
* Configurable model dimensions
* NumPy-only implementation

## Architecture

The runtime consists of the following major components:

```text
                    ┌────────────────────┐
                    │   RuntimeScheduler │
                    └─────────┬──────────┘
                              │
                     ┌────────▼────────┐
                     │   ModelLoader   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  ModelWeights   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────────┐
                     │ TransformerEngine   │
                     └───────┬─────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
       ┌──────▼───────┐              ┌──────▼─────┐
       │ Attention    │              │    FFN     │
       │   Engine     │              │   Engine   │
       └──────┬───────┘              └────────────┘
              │
       ┌──────┴───────────┐
       │                  │
┌──────▼────────┐  ┌──────▼──────────┐
│ GEMM Attention│  │ Flash Attention │
└───────────────┘  └─────────────────┘
              │
       ┌──────▼───────┐
       │ TokenGenerator│
       └───────────────┘
```

## Model Configuration

The Transformer configuration is controlled through the `Config` dataclass.

Example configuration used in `main()`:

| Parameter               |          Value |
| ----------------------- | -------------: |
| Vocabulary size         |         32,000 |
| Hidden dimension        |            256 |
| Attention heads         |              8 |
| Transformer layers      |              4 |
| FFN dimension           |          1,024 |
| Maximum sequence length |             64 |
| Data type               |           FP16 |
| Attention kernel        | FlashAttention |
| Maximum new tokens      |             16 |
| Random seed             |             42 |

The attention head dimension is:

```text
head_dim = hidden_dim / number_of_heads
         = 256 / 8
         = 32
```

## Transformer Architecture

Each Transformer layer follows the structure:

```text
Input
  │
  ▼
Layer Normalization
  │
  ▼
Multi-Head Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Feed Forward Network
  │
  ▼
Residual Connection
  │
  ▼
Output
```

The implementation uses:

* Query projection (`Wq`)
* Key projection (`Wk`)
* Value projection (`Wv`)
* Output projection (`Wo`)
* FFN projection (`W1`)
* FFN projection (`W2`)
* Layer normalization
* Residual connections

## Attention

The attention engine supports two execution paths.

### GEMM-Based Attention

The standard attention calculation is:

```text
Attention(Q,K,V) =
softmax(QKᵀ / √d) V
```

The implementation computes the attention scores using NumPy `einsum`.

This represents a straightforward `O(T²)` attention implementation.

### FlashAttention-Style Attention

The project also contains a simulated block-wise FlashAttention implementation.

Instead of constructing the complete attention matrix at once, keys and values are processed in blocks:

```text
Q
│
├── K/V Block 1
├── K/V Block 2
├── K/V Block 3
├── ...
└── K/V Block N
```

The implementation maintains:

* Running row maximum
* Running normalization sum
* Partial output

This models the main idea of **streaming/block-wise attention with reduced intermediate memory traffic**.

> This is a simulation of FlashAttention behavior using NumPy; it is not the CUDA FlashAttention kernel.

## Precision and Quantization

The `Quantizer` class supports three modes.

### FP32

```text
FP32 → 32-bit floating point
```

### FP16

The values are converted to FP16 and then back to FP32:

```text
FP32 → FP16 → FP32
```

This simulates reduced-precision numerical effects.

### INT8

The implementation performs symmetric-style scaling:

```text
scale = max(|x|) / 127
```

Values are quantized using:

```text
q = round(x / scale)
```

and reconstructed using:

```text
x ≈ q × scale
```

The implementation explicitly describes this as a **simulation**, not real INT8 hardware bit-packing.

## KV Cache

The Transformer runtime maintains a KV cache for each Transformer layer.

During autoregressive generation:

```text
Previous K,V
     │
     ▼
KV Cache ──────┐
               │
New K,V ───────┤
               ▼
         Updated Cache
```

The `MemoryManager` tracks the growth of the KV cache as new tokens are generated.

This is important for understanding LLM inference because autoregressive decoding repeatedly reuses previously computed keys and values.

## Memory Manager

The `MemoryManager` tracks simulated allocations for:

* Token embeddings
* Attention weights
* FFN weights
* Output projection
* KV cache

Supported data types have the following simulated element sizes:

```text
FP32 → 4 bytes
FP16 → 2 bytes
INT8 → 1 byte
```

The runtime produces a memory report showing the simulated memory consumption of each component.

## Performance Profiler

The `Profiler` measures wall-clock execution time for different runtime stages.

Tracked stages include:

* Model loading
* Embedding
* GEMM attention
* FlashAttention
* Layer operations
* FFN
* Output projection
* Total generation

The profiler produces a report containing:

```text
Stage                  Calls    Time(ms)    Share
--------------------------------------------------
...
TOTAL
```

This makes the project useful for experimenting with **Transformer performance analysis and runtime bottlenecks**.

## Token Generation

The `TokenGenerator` performs autoregressive greedy decoding.

The process is:

```text
Prompt
  │
  ▼
Transformer Forward Pass
  │
  ▼
Last Hidden State
  │
  ▼
Output Projection
  │
  ▼
Logits
  │
  ▼
argmax
  │
  ▼
Next Token
  │
  └──────────────► Repeat
```

The next token is selected using:

```python
next_id = int(np.argmax(logits))
```

Therefore, the current implementation uses **greedy decoding**, rather than temperature sampling, top-k, or top-p sampling.

## Model Weights

The model weights are generated randomly using NumPy's random number generator.

The model contains:

```text
Embedding
    │
    ├── Transformer Layer 1
    ├── Transformer Layer 2
    ├── Transformer Layer 3
    └── Transformer Layer 4
    │
Output Projection
```

The weights are initialized using a normal distribution scaled by `0.02`.

Because the model is **not trained**, the generated token IDs do not represent meaningful natural-language text.

## Requirements

The notebook requires Python and NumPy.

Install NumPy with:

```bash
pip install numpy
```

For Jupyter Notebook:

```bash
pip install jupyter numpy
```

## Running the Project

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Transformer_accipynb.ipynb
```

Run the cells in order.

The `main()` function creates the model configuration, generates a random prompt, runs inference, and prints:

1. Configuration
2. Input token IDs
3. Generated token IDs
4. Memory report
5. Performance report

## Example Workflow

```python
cfg = Config(
    vocab_size=32000,
    hidden_dim=256,
    n_heads=8,
    n_layers=4,
    ffn_dim=1024,
    max_seq_len=64,
    dtype="fp16",
    attention_kernel="flash",
    max_new_tokens=16,
    seed=42,
)
```

A random prompt is then generated:

```python
prompt_ids = rng.integers(
    0,
    cfg.vocab_size,
    size=8
).tolist()
```

The runtime executes:

```python
result = scheduler.run(prompt_ids)
```

and returns:

```python
{
    "input_ids": ...,
    "output_ids": ...,
    "generated_ids": ...
}
```

## Project Structure

The notebook can conceptually be organized as:

```text
Transformer Runtime
│
├── Config
│   └── Model/runtime configuration
│
├── Profiler
│   └── Runtime performance measurement
│
├── MemoryManager
│   └── Weight and KV-cache tracking
│
├── Quantizer
│   └── FP32 / FP16 / INT8 simulation
│
├── ModelWeights
│   └── Transformer parameters
│
├── ModelLoader
│   └── Weight initialization/loading
│
├── AttentionEngine
│   ├── GEMM Attention
│   └── FlashAttention simulation
│
├── FFNEngine
│   └── Feed-forward computation
│
├── TransformerEngine
│   └── Complete Transformer forward pass
│
├── TokenGenerator
│   └── Autoregressive decoding
│
└── RuntimeScheduler
    └── End-to-end inference orchestration
```

## What This Project Demonstrates

This project is particularly useful for studying the **systems side of LLM inference**.

It demonstrates how different components interact:

```text
Model Architecture
       ↓
Numerical Precision
       ↓
Attention Kernel
       ↓
KV Cache
       ↓
Memory Usage
       ↓
Runtime Performance
       ↓
Token Generation
```

It can therefore serve as a foundation for experimenting with **AI systems, GPU/NPU architecture, Transformer optimization, and inference-runtime design**.

## Limitations

The current implementation is intentionally a simulation and has several limitations:

* Model weights are randomly initialized.
* There is no training pipeline.
* There is no tokenizer.
* Generated IDs cannot be decoded into meaningful text.
* Attention is implemented with NumPy rather than CUDA.
* FlashAttention is simulated rather than using the actual optimized CUDA kernel.
* INT8 is simulated numerically and does not perform hardware-level packed INT8 computation.
* There is no GPU execution.
* There is no distributed inference.
* There is no real model checkpoint loading.
* The FFN uses ReLU rather than the activation functions commonly used in modern LLMs.
* The implementation is primarily intended for runtime/architecture experimentation rather than production inference.

## Possible Extensions

Future versions could add:

### 1. Real Tokenizer

Integrate a tokenizer such as BPE or SentencePiece:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Transformer
 ↓
Token IDs
 ↓
Tokenizer
 ↓
Text
```

### 2. Real Model Weights

Load a pretrained Transformer checkpoint instead of randomly generated weights.

### 3. GPU Support

Replace NumPy operations with:

* PyTorch
* CUDA
* Triton
* CuPy

### 4. Real FlashAttention

Integrate an optimized FlashAttention implementation and compare it against the naive attention path.

### 5. Real INT8 Inference

Implement:

* Weight-only quantization
* Activation quantization
* INT8 GEMM
* Calibration
* Per-channel/per-group scaling

### 6. Hardware-Aware Analysis

The runtime could be extended to model:

```text
Transformer
    ↓
Operator Graph
    ↓
Memory Traffic
    ↓
Compute Requirements
    ↓
GPU/NPU Mapping
    ↓
Latency / Throughput
    ↓
Energy
```

### 7. Benchmarking

Run experiments comparing:

```text
FP32 vs FP16 vs INT8

GEMM Attention vs FlashAttention

Different sequence lengths

Different hidden dimensions

Different numbers of attention heads

Different numbers of Transformer layers
```

## Research Direction

A natural extension of this project is a **Transformer inference hardware/software co-design simulator**.

For example:

```text
             Transformer Model
                    │
        ┌───────────┴───────────┐
        │                       │
   Model Analysis          Quantization
        │                       │
        └───────────┬───────────┘
                    │
             Operator Mapping
                    │
        ┌───────────┴───────────┐
        │                       │
       GPU                     NPU
        │                       │
        └───────────┬───────────┘
                    │
          Performance Model
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     Latency     Memory      Energy
```

This would make the project relevant to **LLM inference optimization, AI accelerators, NPU architecture, and computer architecture research**.

## License

Add the license appropriate for your project, for example:

```text
MIT License
```

## Author

**Anoop**

Transformer runtime and inference simulation project focused on understanding Transformer execution, attention optimization, quantization, memory management, and runtime performance.
