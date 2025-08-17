# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning visualization project focused on understanding training dynamics and model behavior. The project includes neural network training experiments (particularly on MNIST), clustering algorithms, and visualization tools with cloud storage integration.

## Development Commands

### Python Environment
- Uses `uv` package manager with `pyproject.toml` configuration
- Install dependencies: `uv sync`
- Run Python scripts: `python <script_path>`

### Testing Configuration
- Test configuration validation: `python -c "from shared_lib.r2 import _validate_config; _validate_config(); print('✅ Configuration is valid!')"`
- Run tests: `python test-scripts/test_media.py`

### Environment Setup
- Copy `.env.example` to `.env` and configure Cloudflare R2 credentials
- Required environment variables: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`

## Code Architecture

### Core Components

**shared_lib/** - Shared utilities and core functionality:
- `datasets.py` - Dataset loading with caching (MNIST, Fashion-MNIST, CIFAR-10) using HuggingFace datasets
- `r2.py` - Cloudflare R2 storage integration with AWS Signature V4 authentication
- `media.py` - Media handling with R2 upload fallback to local storage
- `random_utils.py` - Random number generation utilities

**projects/** - Experiment implementations:
- `online/` - Online learning experiments with visualization
- `kmeans/` - K-means clustering experiments

### Key Patterns

**Dataset Loading**: Uses `load_supervised_image()` and `load_supervised_1d()` functions that return typed NamedTuples (`ImageClassification`, `Supervised1D`) with standardized fields for training/test data.

**JAX-based ML**: Uses JAX for neural networks with:
- `init()` function that returns (config, params, key_gen) tuple for model initialization
- `train_step()` JIT-compiled training functions
- Optax optimizers (typically Adam)

**Visualization & Storage**: 
- Matplotlib figures saved via `save_matplotlib_figure()`
- Media files handled through `save_media()` with R2/local fallback
- Animation generation for training dynamics

**Training Evaluation**: Comprehensive evaluation system in `online_eval.py`:
- Section-based accuracy analysis
- Permutation-aware evaluation
- Training progress animation generation

### Data Flow

1. Load datasets using `shared_lib.datasets`
2. Initialize JAX model parameters and optimizers
3. Train with visualization collection at regular intervals
4. Save results to R2 (cloud) or local outputs/ directory
5. Generate plots and animations showing training dynamics

## File Organization

- `outputs/` - Generated files organized by date (yy-mm-dd format)
- `mermaids/` - Mermaid diagram files
- `test-scripts/` - Test utilities
- Projects are self-contained in `projects/` subdirectories