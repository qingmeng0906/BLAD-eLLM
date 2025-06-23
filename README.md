# BLAD-eLLM: Blockchain Live Anomaly Detection with eBPF and LLMs
BLAD-eLLM is an framework for real-time anomaly detection in blockchain systems, integrating extended Berkeley Packet Filter(eBPF) technology and Large Language Models (LLMs). This repository contains the implementation of our novel approach for monitoring, detecting, and mitigating malicious activities in blockchain networks.


## 🧠 Anomaly Detection Module

The core of our system is the anomaly detection module, which leverages fine-tuned Large Language Models to identify suspicious blockchain activities.
### Key Features
#### 🎯 Chain-of-Thought (CoT) Reasoning
Our implementation incorporates CoT prompts that guide the LLM through step-by-step reasoning:
1. **Field Analysis**: Understanding the meaning of each data field
2. **Pattern Recognition**: Identifying anomalous patterns in traffic
3. **Comparison**: Benchmarking against normal traffic models
4. **Decision Synthesis**: Generating final anomaly classification

#### 🛡️ Security Shield
To defend against prompt injection attacks, we implement an Ignore Defense mechanism:
- Instructs the model to ignore potential injection instructions
- Re-emphasizes original task objectives
- Maintains model focus on anomaly detection

#### 📚 Retrieval-Augmented Generation (RAG)
The system employs a BERT-based RAG framework:
- Retrieves relevant information from curated knowledge bases
- Enhances LLM decisions with verified external knowledge
- Improves detection accuracy through contextual understanding

### 🚀 Model Fine-tuning

The module fine-tunes the Qwen2.5-7B model using:
- **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Efficient adaptation while preserving pre-trained weights
- **Domain-specific prompts**: Blockchain security-focused training data
- **Enhanced reasoning capabilities**: CoT-guided decision making

## ⚠️ Current Status

**Note**: This initial release contains a part of components of the anomaly detection framework. The complete implementation with full optimizations will be released in the future as part of our research publication.
