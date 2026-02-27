# 🍎 LLM Inference on Apple Silicon with MLX

> **Building and optimizing LLM inference from scratch — no cloud, no CUDA, just your Mac.**

This repository is the companion codebase for a YouTube series that takes you deep into the world of **Large Language Model (LLM) inference** — how it works, what makes it slow, and how to make it fast — entirely on **Apple Silicon** using Apple's own ML framework, [**MLX**](https://github.com/ml-explore/mlx).

We skip the hand-waving. We skip the vague explanations. Every concept is grounded in code you can run yourself.

---

## 📺 Series Overview

| Part | Title | Status |
|------|-------|--------|
| **Part 1** | What Actually Happens When You Type a Prompt? | ✅ Released |
| **Part 2** | Setting Up MLX & Running Your First Inference on Apple Silicon | 🔜 Coming Soon |
| **Part 3** | Understanding the KV Cache — And Building It in Code | 🔜 Coming Soon |
| **Part 4** | Quantization — Making Models Smaller Without Breaking Them | 🔜 Coming Soon |
| **Part 5** | Batching, Throughput & Latency — The Real Bottlenecks | 🔜 Coming Soon |
| **Part 6** | Building a Full Inference Pipeline from Scratch | 🔜 Coming Soon |

---

## 🗺️ Part 1 — The Complete Journey of a Prompt

Before a single line of code, you need to understand what's actually happening when you type something into ChatGPT and hit enter. This is the full pipeline:

```
You Type a Prompt
        │
        ▼
┌───────────────┐
│  Tokenizer    │  Converts text → Token IDs (integers)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Embedding    │  Maps Token IDs → Dense Vectors
│  Layer        │  (Token Embedding + Positional Embedding)
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────┐
│         Transformer Blocks        │
│                                   │
│   ┌─────────────────────────┐     │
│   │     Self-Attention       │     │  ← Runs across N layers
│   │  (Q · Kᵀ / √dₖ) · V    │     │
│   └────────────┬────────────┘     │
│                │                  │
│   ┌────────────▼────────────┐     │
│   │   Feed Forward Network  │     │
│   └─────────────────────────┘     │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────┐
│  Linear + Softmax     │  Produces probability distribution
│  (over full vocab)    │  over next token
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Sampling             │  Pick next token (greedy / top-k / temperature)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Autoregressive Loop  │  Append token → repeat from Transformer
│  (+ KV Cache)         │  until <EOS> or max tokens
└──────────┬────────────┘
           │
           ▼
    Stream to Screen ✅
```<img width="308" height="836" alt="Screenshot 2026-02-27 at 10 31 09 AM" src="https://github.com/user-attachments/assets/6b33965b-fa27-47b9-a7eb-33d5e26519b5" />


---

### Step 1 — Tokenization

The model cannot understand English, Hindi, or any human language. It only operates on **numbers**. The tokenizer's job is to convert your text into a sequence of integer IDs.

Tokens are **not** simply words. They can be full words, subwords, punctuation, or even whitespace — depending on the tokenizer's vocabulary, built using **Byte Pair Encoding (BPE)**.

```
Input:  "Explain black holes to me"
Output: [9004, 3543, 10349, 284, 502]

Examples:
  "cat"           →  [1 token]
  "unbelievable"  →  ["un"] ["believ"] ["able"]  →  [3 tokens]
  "ChatGPT"       →  ["Chat"] ["G"] ["PT"]        →  [3 tokens]
  " hello"        →  [1 token]  ← note: the space is included!
```

> ⚠️ Token IDs are just **index numbers** in a lookup table. The number `9004` is not semantically "bigger" than `502` — they're arbitrary labels. This is why we need embeddings next.

---

### Step 2 — The Embedding Layer

A neural network needs numbers it can **do meaningful math on**. Raw token IDs carry no semantic information — `King=776` and `Man=1` have no implied relationship from their IDs alone.

The **Embedding Layer** maps each Token ID to a dense **vector** — a list of floating point numbers representing a coordinate in high-dimensional space.

**Key idea:** Words used in similar contexts end up with vectors that are **close together** in this space. The distance between vectors is meaningful.

#### 🍕 Intuitive Example — Food Embeddings

Imagine representing foods on two axes:

```
             HOT
              │
   Pizza ─────┼───── Soup
              │
SAVORY ───────┼───────────── SWEET
              │
   Chips ─────┼───── Ice Cream
              │
             COLD
```

- 🍕 **Pizza** and 🍲 **Soup** cluster together → similar vectors
- 🍦 **Ice Cream** and 🥤 **Soda** cluster together → similar vectors  
- 🍕 **Pizza** and 🍦 **Ice Cream** are far apart → very different vectors

In practice, GPT-2 uses **768 dimensions** — not 2. More dimensions = more capacity to capture nuanced meaning, but also **more memory and slower computation**. This tradeoff is at the heart of inference optimization.

#### Positional Embedding

The token vector encodes *what* a word means — but not *where* it sits in the sentence. Position matters: "The dog bit the man" ≠ "The man bit the dog."

The embedding layer generates a **Positional Embedding** alongside the token embedding, and the two are combined:

```
Final Vector = Token Embedding + Positional Embedding
```

This final vector is what flows into the Transformer.

---

### Step 3 — Self-Attention

Consider these two sentences:

```
"The Crane ate a fish."
"The Crane lifted the steel."
```

At the embedding stage, **Crane** gets the exact same vector in both sentences — the lookup table has no context. But they clearly refer to different things: a bird vs. a machine.

**Self-Attention** solves this by asking:

> *For each token in the sequence — how much should it pay attention to every other token?*

It updates each token's vector based on its neighbors, allowing the same word to have **completely different representations** depending on context.

#### The QKV Mechanism

Self-Attention uses three learned projections — **Queries (Q)**, **Keys (K)**, and **Values (V)**:

| Component | Analogy | Role |
|-----------|---------|------|
| **Q (Query)** | "What am I looking for?" | The current token asking about context |
| **K (Key)** | "What do I have to offer?" | All other tokens advertising their relevance |
| **V (Value)** | "What information do I pass forward?" | The actual content retrieved |

The attention score between a Query and all Keys determines how much each Value contributes to the updated representation.

#### The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $QK^T$ — dot product between Query and all Keys (measures similarity)
- $\sqrt{d_k}$ — scaling factor (prevents dot products from getting too large and saturating softmax)
- $\text{softmax}(\cdot)$ — converts raw scores into a probability distribution (attention weights)
- Multiplying by $V$ — weighted sum of Values using those attention weights

**Result:** The Crane vector in sentence 1 gets updated toward *"ate"* and *"fish"* → reads as a bird.
The Crane vector in sentence 2 gets updated toward *"lifted"* and *"steel"* → reads as a machine.

This entire Attention + Feed Forward block is stacked **N times** (e.g., GPT-2 has 12 layers, GPT-3 has 96). Each layer refines the representation further.

---

### Step 4 — Predicting the Next Token

After the final Transformer layer, the output passes through:

1. **Linear Layer** — projects the vector into vocabulary size space
2. **Softmax** — converts raw scores into a probability distribution over all tokens

```
Given: "Explain black holes"

Next token probabilities:
  "are"    ████████████  42%
  "is"     ██████        18%
  "have"   ████          12%
  "exist"  ███            9%
  "were"   ██             6%
  ...
```

The model then **samples** from this distribution to pick the next token. This is controlled by **temperature**:

```
Temperature = 0.2  →  Conservative, deterministic, "plays it safe"
Temperature = 1.0  →  Balanced
Temperature = 1.5  →  Creative, surprising, occasionally chaotic
```

---

### Step 5 — Autoregressive Decoding

LLMs do **not** generate the full response at once. They generate **one token at a time** in a loop:

```
Round 1: "Explain black holes"              → predicts "are"
Round 2: "Explain black holes are"          → predicts "regions"
Round 3: "Explain black holes are regions"  → predicts "of"
Round 4: ...                                → continues until <EOS>
```

This is **Autoregressive Decoding** — each generated token is appended to the input and fed back into the model. The text streaming to your screen is this loop happening in real time.

---

### Step 6 — The KV Cache Problem (And Why It Matters)

Inside each autoregressive step, Self-Attention needs the **Key and Value vectors** for every previous token.

**The naive approach:** Recompute K and V for all previous tokens from scratch on every single step.

```
Generating token 200?
→ Recompute K, V for tokens 1–199.   ← Every. Single. Time.
```

This is enormously expensive. For a 200-token response, you'd be doing nearly 20,000 redundant matrix multiplications.

**The KV Cache solution:** Compute K and V once per token and **store them in memory**. On each new step, only compute K and V for the **new token** and reuse everything cached.

```
Step 1: Compute K, V for "The"  → store in cache ✅
Step 2: Compute K, V for "Cat"  → store in cache ✅
Step 3: Only compute K, V for new token "Sat"
        Reuse "The" and "Cat" from cache ✅
```

#### The Memory Tradeoff

The KV Cache lives in **RAM**. As sequences get longer, the cache grows linearly. Managing it efficiently — what to keep, when to evict, how to lay it out — is one of the central challenges of LLM inference optimization.

This is **exactly** where Apple Silicon has a significant architectural advantage due to its **Unified Memory Architecture (UMA)** — which we'll explore hands-on in Part 2.

---

## 🍎 Why Apple Silicon + MLX?

| | PyTorch (CUDA) | MLX (Apple Silicon) |
|--|----------------|---------------------|
| Memory | Separate CPU & GPU RAM | **Unified Memory** — CPU & GPU share the same pool |
| Framework | NVIDIA-specific | Apple-native, built for M-series chips |
| Quantization | External tools needed | First-class support in MLX |
| Power efficiency | High power draw | Designed for efficiency |
| Local inference | Requires expensive GPU | **Runs on your MacBook** |

In standard GPU systems, data must be explicitly copied between CPU RAM and GPU VRAM. On Apple Silicon, the CPU, GPU, and Neural Engine all share **one memory pool** — eliminating that bottleneck entirely. For LLM inference, where memory bandwidth is often the primary bottleneck, this is a game changer.

---

## 🚀 Getting Started

### Prerequisites

- Mac with Apple Silicon (M1, M2, M3, or M4 chip)
- Python 3.10+
- Homebrew (recommended)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/llm-inference-mlx.git
cd llm-inference-mlx

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install MLX
pip install mlx mlx-lm
```

### Run Your First Inference

```bash
python inference.py --model mlx-community/Mistral-7B-v0.1-4bit-MLX --prompt "Explain black holes"
```

---

## 📁 Repository Structure

```
llm-inference-mlx/
│
├── part1/                  # Prompt journey — concepts & diagrams
│   └── walkthrough.md
│
├── part2/                  # MLX setup & first inference
│   ├── inference.py
│   └── README.md
│
├── part3/                  # KV Cache — explanation + implementation
│   ├── kv_cache.py
│   └── README.md
│
├── part4/                  # Quantization
│   ├── quantize.py
│   └── README.md
│
├── assets/                 # Diagrams and images used in videos
│
├── requirements.txt
└── README.md               ← You are here
```

---

## 📐 Key Formulas Reference

### Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

### KV Cache Memory Cost

$$\text{KV Cache Size} = 2 \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times S_{\text{seq}} \times \text{dtype\_bytes}$$

Where:
- $N_{\text{layers}}$ = number of transformer layers
- $N_{\text{heads}}$ = number of attention heads
- $d_{\text{head}}$ = dimension per head
- $S_{\text{seq}}$ = current sequence length
- Factor of 2 accounts for both K and V

### Softmax (Token Sampling)

$$\text{softmax}(z_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

Where $T$ is **temperature** — higher $T$ flattens the distribution (more creative), lower $T$ sharpens it (more deterministic).

---

## 🤝 Contributing

This series is all about learning in public. If you spot a bug, a technical inaccuracy, or have a suggestion — open an issue or PR. All levels welcome.

---

## 📬 Connect

- 🎥 YouTube: [your channel link]
- 🐦 Twitter/X: [@yourhandle]
- 💼 LinkedIn: [your profile]

---

## 📄 License

MIT License — use the code freely, build on it, and share what you learn.

---

> *"The best way to understand a system is to build it yourself."*
