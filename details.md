# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Uncertainty-aware Adaptive Guidance (UAG)**, a research project for enhancing Large Language Model reasoning through uncertainty-based adaptive guidance. The codebase is from the ACL 2024 paper "Reasoning in Flux: Enhancing Large Language Models Reasoning through Uncertainty-aware Adaptive Guidance".

### Research Motivation

The paper addresses a critical issue in LLM reasoning: **error propagation and accumulation** in multi-step reasoning chains. Traditional approaches like Chain-of-Thought (CoT) can suffer from reasoning errors that compound throughout the process. UAG provides a principled approach to detect and correct reasoning errors before they propagate.

### Key Research Contributions

1. **Uncertainty Quantification**: Novel approach to measure reasoning uncertainty at the token level using negative log probabilities
2. **Adaptive Intervention**: Dynamic guidance system that intervenes only when uncertainty exceeds a threshold
3. **Demonstration Selection**: Smart retrieval of relevant examples based on both relevance and originality
4. **Error Prevention**: Proactive correction mechanism that prevents error accumulation

## Core Architecture

The main implementation follows a modular design:

- **`uag.py`**: Core UAG class implementing the uncertainty-aware adaptive guidance algorithm
- **`prompt.py`**: Task-specific prompts and demonstrations for GSM8K, AQuA, CSQA, and StrategyQA
- **`metric.py`**: Evaluation metrics for different reasoning tasks
- **`main.py`**: Simple entry point script

### Key Components

1. **UAG Class** (`uag.py`): 
   - Uses Mistral-7B-Instruct-v0.3 as the main reasoning model
   - Uses nvidia/NV-Embed-v2 for embedding computations
   - Implements uncertainty computation, demonstration clustering, and adaptive reasoning adjustment

2. **Uncertainty Mechanism**:
   - Computes token-level uncertainties using negative log probabilities: `u_i = -log P(x_i | x_<i)`
   - Uses delta uncertainties (uncertainty differences between consecutive tokens): `Δu_i = u_i - u_{i-1}`
   - Threshold `theta` determines when to trigger adaptive guidance
   - **Theoretical Foundation**: Based on information theory - higher uncertainty indicates model confusion

3. **Demonstration System**:
   - Clusters demonstrations using K-means on embeddings into k clusters
   - **Selection Score**: `S(D) = λ₁ × R(D) + λ₂ × O(D)` where:
     - `R(D)`: Relevance score = `log P(D | Q, r_≤m)` (likelihood of demonstration given question and partial reasoning)
     - `O(D)`: Originality score = `-log P(D | Q)` (inverse likelihood given only question)
   - Weighted by `lambda1` (relevance) and `lambda2` (originality)
   - **Purpose**: Balance between contextually relevant and novel/diverse demonstrations

## Running the Code

### Basic Usage

```bash
python code/uag.py --task GSM8K --data-path GSM8K_input.jsonl --record-path GSM8K_output.jsonl
```

### Key Parameters

- `--theta`: Uncertainty threshold (default: 16) - controls when UAG intervenes
- `--lambda1`: Weight for relevance score (default: 0.5)
- `--lambda2`: Weight for originality score (default: 0.5)
- `--k`: Number of demonstration clusters (default: 8)
- `--temperature`: Generation temperature (default: 0.5)
- `--max-length`: Maximum sequence length (default: 2048)
- `--max-loop`: Maximum adjustment loops (default: 10)

### Supported Tasks

- **GSM8K**: Grade school math problems
- **AQuA**: Algebraic word problems with multiple choice
- **CSQA**: CommonsenseQA multiple choice reasoning
- **StrategyQA**: Yes/no strategy questions

## Dependencies

Based on README.md, required packages:
- transformers >= 4.46.2
- torch
- numpy
- sklearn
- tqdm

## Data Format

Input data should be JSONL format with:
- `question`: The reasoning question
- `answer`: Ground truth answer (optional for inference)

Output includes:
- Original question and answer
- Generated reasoning chain
- Predicted answer
- Correctness evaluation
- Uncertainty measurements

## Algorithm Details

### UAG Workflow (from paper)

1. **Initial Generation**: Generate reasoning chain using standard prompting
2. **Uncertainty Monitoring**: Compute `Δu_i` for each reasoning step
3. **Intervention Decision**: If `Δu_i > θ`, trigger adaptive guidance
4. **Backtracking**: Remove reasoning steps back to last reliable point (step token `\n`)
5. **Demonstration Retrieval**: Select most relevant demonstrations from clusters
6. **Guided Generation**: Generate continuation using selected demonstrations
7. **Iteration**: Repeat until uncertainty stays below threshold or max loops reached

### Experimental Results (from paper)

- **GSM8K**: 84.7% accuracy (vs 79.4% baseline)
- **AQuA**: 35.2% accuracy (vs 31.8% baseline)
- **CSQA**: 82.8% accuracy (vs 81.1% baseline)
- **StrategyQA**: 75.6% accuracy (vs 72.9% baseline)

### Hyperparameter Sensitivity

- **θ (theta)**: Paper shows optimal range 12-20 for most tasks
- **λ₁, λ₂**: Equal weighting (0.5, 0.5) works well across tasks
- **k (clusters)**: 8 clusters provide good balance of diversity and relevance

## Development and Verification Strategy

This project follows a two-phase approach to ensure robust replication of the paper's findings.

### Phase 1: Framework Validation (Completed)

The initial phase focused on verifying the integrity of the experimental framework using lightweight, fast-loading models.

- **Objective**: To debug the code, confirm environment compatibility (M1 Mac, Conda), and validate the UAG algorithm's control flow without the overhead of large models.
- **Models Used**:
  - **Reasoning Model**: `distilgpt2`
  - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Key Bugs Fixed**:
  - `FileNotFoundError`: Corrected output directory creation logic.
  - `AttributeError` / `ValueError`: Refactored the embedding generation and data structures to fix crashes within the `adaptive_reasoning_adjustment` function.
- **Outcome**: Successfully created a stable, runnable experimental testbed. The UAG mechanism was programmatically triggered and verified, even though the final accuracy was 0.0, as expected with a non-reasoning model.

### Phase 2: Performance Replication (Next Step)

This phase aims to replicate the performance metrics reported in the paper by using the original, powerful models.

- **Objective**: To achieve accuracy scores comparable to the paper's results (e.g., 84.7% on GSM8K) and validate the effectiveness of the UAG algorithm.
- **Models to Use**:
  - **Reasoning Model**: `mistralai/Mistral-7B-Instruct-v0.3`
  - **Embedding Model**: `nvidia/NV-Embed-v2`

## Development Notes

- The code uses CUDA device 0 by default (hardcoded in `uag.py`)
- Embedding model (nvidia/NV-Embed-v2) requires `trust_remote_code=True`
- Each task has specific prompt formats and answer extraction logic in `metric.py`
- Demonstration clustering happens once during initialization for efficiency
- **Performance**: UAG adds ~2-3x computational overhead compared to standard CoT
- **Memory**: Requires storing demonstration embeddings and model states for backtracking