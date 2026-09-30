# Large Language Models with Safetensors

This example demonstrates how to optimize large language models using the [safetensors](https://github.com/safetensors/safetensors) format.
This is useful when deploying quantized LLMs by quickly switching between different quantization methods, such as GPTQ and AWQ.

## Setup

To get started, clone the Kenning repository:

```bash test-skip
git clone https://github.com/antmicro/kenning.git
cd kenning/
```

and install it with

```bash
uv pip install ".[llm,sparsity-aware-kernel,reports]"
```

{{uses_gpu}}

## AWQ Optimization

The first example shows how to use AWQ optimization to protect important weights in order to maintain model accuracy.
The configuration specifies the `model_framework` as `safetensors` to ensure compatible loading.

Use the following configuration:

```{literalinclude} ../scripts/configs/llm-mistral-awq.yaml save-as=llm-mistral-awq.yaml
:language: yaml
```

Run the optimization flow:

```bash
kenning optimize --cfg llm-mistral-awq.yaml
```

## SparseGPTQ Optimization

Kenning supports [SparseGPTQ](https://arxiv.org/abs/2301.00774) which combines quantization with structured pruning.
It is useful when reduction in both the memory footprint and computational cost is required.
This approach allows the LLM to run on hardware very limited RAM and processing power.

Use the following configuration:

```{literalinclude} ../scripts/configs/llm-mistral-sparsegptq.yaml save-as=llm-mistral-sparsegptq.yaml
:language: yaml
```

Run the optimization flow:

```bash
kenning optimize --cfg llm-mistral-sparsegptq.yaml
```

## vLLM Runtime

The resulting safetensors models can be deployed using the vLLM runtime.
vLLM is required to perform the tests and evaluation.

## Results

First, perform the evaluation for each model:

```bash
kenning test \
  --cfg llm-mistral-awq.yaml \
  --measurements build/awq.json
```

and if you want to evaluate `sparsegpt` tests:

```
kenning test \
  --cfg llm-mistral-sparsegptq.yaml \
  --measurements build/sparsegptq.json
```

Then, generate the comparison report:

```bash
kenning report \
  --measurements \
    build/awq.json \
    build/sparsegptq.json \
  --report-path build/llm-comparison.md \
  --to-html
```
