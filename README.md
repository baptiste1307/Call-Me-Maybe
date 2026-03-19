<p align="center">
  <h1 align="center">Call-Me-Maybe</h1>
</p>

<p align="center">
  <strong>Constrained decoding and function calling engine with local causal LLMs.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#example-usage">Example</a> •
  <a href="#algorithm-explanation">Algorithm</a> •
  <a href="#design-decisions">Design</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic" />
  <img src="https://img.shields.io/badge/Package_Manager-uv-DE5FE9?style=flat-square&logo=astral&logoColor=white" alt="uv" />
  <img src="https://img.shields.io/badge/License-MIT-black?style=flat-square" alt="License" />
</p>

---

## Overview

Call-Me-Maybe implements a high-precision **function calling system powered by local causal language models** (`Qwen/Qwen3-0.6B`). Rather than allowing unconstrained generative output that drifts or hallucinates syntax, the engine uses **token-level constrained decoding** to guarantee that every output strictly matches the target JSON schema and available function signatures.

- **Deterministic Function Routing**: Prefix-tree token filtering enforces valid function selection from raw logits without hallucinated identifiers.
- **Strict Schema Compliance**: Enforces 100% valid JSON serialization matching Pydantic v2 data models.
- **Type-Aware Parameter Extraction**: Parses parameters into typed primitives (`string`, `number`, `boolean`) with rigorous validation.
- **Local & Offline Inference**: Runs entirely on local hardware (CPU, CUDA, Apple Silicon MPS) without third-party API dependencies.

---

## Key Features

- **Constrained Decoding Engine**: Inspects next-token logits and restricts candidate tokens to valid prefixes of known function names, preventing generation of non-existent tools.
- **Pydantic Validation**: Strong runtime schema validation for function signatures, inputs, and structured outputs.
- **Clean Architecture & Separation of Concerns**:
  - `decoder.py`: Token-level filtering, prefix tree checks, and logit ranking.
  - `generator.py`: Prompt routing, function resolution, and parameter binding.
  - `models.py`: Strongly-typed schema definitions using Pydantic.
  - `io_utils.py`: Safe JSON file I/O with clear error messaging.
  - `cli.py`: Command-line interface with customizable input/output paths.
- **Reproducible Environment**: Built and managed with `uv` for ultra-fast dependency resolution and lockfile guarantees.

---

## Quick Start

### 1. Installation

Clone the repository and sync dependencies with `uv`:

```bash
git clone https://github.com/baptiste1307/Call-Me-Maybe.git
cd Call-Me-Maybe
uv sync
```

### 2. Run the Pipeline

Execute the default pipeline:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

If no arguments are provided, default paths in `data/input` and `data/output` are used automatically.

---

## Example Usage

### Input (`data/input/function_calling_tests.json`)

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet shrek"
  }
]
```

### Output (`data/output/function_calling_results.json`)

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  }
]
```

---

## Algorithm Explanation (Constrained Decoding)

Function selection is performed using token-level constrained decoding:

1. **Tokenization**: All candidate function names are tokenized with the model tokenizer.
2. **Prefix Matching**: At each generation step, candidate continuations are filtered against legal prefixes (`allowed_next_tokens`).
3. **Logit Masking & Selection**: The candidate token with the highest logit among valid continuations is picked (`find_best_token`).
4. **Termination**: Generation completes once a valid function identifier is fully resolved.

This mathematical constraint guarantees that:
- Non-existent function names cannot be produced.
- The generation is deterministic and resistant to prompt injection.

---

## Design Decisions

- **Pydantic v2 Models**: Strict validation with `model_config = {"extra": "forbid"}`.
- **Separation of Concerns**: Decoupled modules for decoding, schema modeling, pipeline generation, and CLI.
- **Dynamic Device Support**: Automatically selects Apple Silicon (`mps`), NVIDIA (`cuda`), or `cpu` based on hardware availability.
- **Defensive Error Handling**: Missing files, schema mismatches, and unsupported prompts exit cleanly with clear messages.

---

## Quality & Verification

```bash
# Code style check
uv run flake8 src

# Strict type checking
uv run mypy src
```

---

## License

Distributed under the [MIT License](LICENSE). Built with ❤️ by [baptiste1307](https://github.com/baptiste1307).
