# Fine-Tuning Tools Comparison: PEFT vs LLaMA-Factory vs Unsloth vs LitGPT

This project fine-tunes Qwen2.5-1.5B-Instruct using **LoRA**. LoRA itself is
a *technique*; the tools below are different ways to actually run it. This
doc compares them and explains why **LLaMA-Factory** was chosen here.

| Tool | Level | Best for | Main limitation | GitHub |
|---|---|---|---|---|
| **PEFT** | Low-level library | Full control, custom training loops, research | You write everything yourself (training loop, data handling) | https://github.com/huggingface/peft |
| **LLaMA-Factory** | High-level framework | Widest model support + widest range of training methods (SFT, DPO, PPO, RM), config-driven (YAML), no custom code needed | Slower / more memory-hungry than Unsloth for the same job | https://github.com/hiyouga/LLaMA-Factory |
| **Unsloth** | Optimized framework | Fastest training + lowest VRAM use on limited GPUs (e.g. free Colab T4) | Supports a narrower set of model families | https://github.com/unslothai/unsloth |
| **LitGPT** | Educational / research framework | Clean, from-scratch implementations; full pretraining support, not just fine-tuning | Smaller community, less "plug-and-play" for arbitrary HF models | https://github.com/Lightning-AI/litgpt |

## What each one actually is

### PEFT (Hugging Face)
The foundational library implementing parameter-efficient fine-tuning
methods — LoRA, QLoRA, Prefix Tuning, etc. — as building blocks you wrap
around a `transformers` model. Every other tool in this table either uses
PEFT internally or reimplements the same ideas. Use it directly when you
need full control over the training loop or are doing something
non-standard that a config file can't express.

### LLaMA-Factory
A framework built on top of PEFT/`transformers` that replaces custom code
with a **YAML config file**. It supports a very wide range of models and
training methods (SFT, reward modeling, PPO, DPO...), plus a WebUI and
built-in experiment tracking integrations (wandb, etc.). Best when you want
to experiment across different models/methods without writing training
code from scratch.

### Unsloth
Focuses purely on speed and memory efficiency: it reimplements the core
LoRA/QLoRA operations as custom Triton kernels, giving **2-5x faster
training and significantly lower VRAM usage** than a plain
`transformers` + PEFT setup, with no change in final accuracy. Ideal for
constrained hardware (e.g. a free Colab GPU) or fast iteration. Trade-off:
it only supports a defined list of model families (Llama, Qwen, Mistral,
Gemma, etc.), not everything.

### LitGPT (Lightning AI)
Prioritizes clean, dependency-light, "no abstractions" code — each model is
implemented from scratch for readability. Unlike the others, it also
supports **full pretraining**, not just fine-tuning an existing checkpoint.
Best when the goal is understanding how a model works internally, or when
pretraining from scratch is actually needed.

## Why LLaMA-Factory was chosen for this project

- **Config over code:** the entire training setup (model, LoRA rank, epochs,
  dataset paths, logging) lives in one YAML file — easy to read, edit, and
  reason about while learning.
- **Qwen2.5 support out of the box** with no extra setup.
- **Built-in wandb integration**, so training progress (loss curves, etc.)
  is visible with almost no extra code.
- **Good fit for a first fine-tuning project**: less code to debug than
  writing a raw PEFT training loop, while still being transparent about
  what's happening (unlike a fully black-box no-code tool).

## When to reach for the others instead

- **Training is too slow or hits Out-Of-Memory on the free GPU** → switch to
  **Unsloth** for the same LoRA fine-tuning job, faster and lighter.
- **Need full control, a custom training loop, or a non-standard setup** →
  drop down to **PEFT** directly.
- **Need to pretrain a model from scratch, or want to study model internals
  in detail** → **LitGPT**.
