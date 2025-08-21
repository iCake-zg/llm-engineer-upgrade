# llm-engineer-upgrade
This repository is dedicated to providing resources, tools, and implementations for enhancing skills in Large Language Models (LLMs) and related fields. It includes projects and solutions that tackle various aspects of LLMs, such as fine-tuning, performance optimization, and deployment strategies.



## rope-handmake
RoPE-Handmade

This repository provides a from-scratch implementation of Rotary Position Embedding (RoPE), inspired by the paper RoFormer: Enhanced Transformer with Rotary Position Embedding
.
It is intended as a hands-on learning project to understand the transition from traditional positional encoding to RoPE.

### 📂 Repository Structure
```bash
rope-handmake/
│
├── 01_ref/                # Reference materials
│   └── 2104.09864v5.pdf   # RoFormer paper
│
├── 02_origina-pe-core/    # Original positional encoding implementation
│   └── pe.py
│
├── 03_rope-core/          # Core RoPE implementation
│   └── rope.py
│
├── 04_debug/              # Debugging and test scripts
│   └── test-rope.py
```
### 🚀 Features

Step-by-step comparison between sinusoidal positional encoding and rotary position embedding.

Minimal implementation for clarity, without external dependencies.

Test scripts for validating the correctness of RoPE.

### 🔧 Usage
-  Run the original positional encoding
python 02_origina-pe-core/pe.py

-  Run the RoPE implementation
python 03_rope-core/rope.py

-  Debug & test
python 04_debug/test-rope.py

### 📖 Reference

RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)



