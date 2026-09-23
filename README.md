# Arabic News Fine-Tuning Project — Overview

## Goal
Fine-tune **Qwen2.5-1.5B-Instruct**, a small open-source LLM, so it can reliably
perform two structured tasks on Arabic news articles:

1. **Details Extraction** — read a news story and extract structured details
   (title, keywords, summary, category, entities) as JSON.
2. **Translation** — translate the story (title + content) into another
   language, also as JSON.

The base model is small (1.5B parameters) and, before training, tends to
produce malformed JSON, invalid categories, or mixed-language output. The
project measures this baseline, fine-tunes the model with **LoRA**, and
compares the result.

## Environment
- **Platform:** Google Colab (free T4 GPU)
- **Training framework:** [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) — only ~1.18% of the
  model's parameters (~18.5M out of ~1.56B) are trained, keeping training
  feasible on a free GPU.
- **Experiment tracking:** [Weights & Biases](https://wandb.ai)
- **Model hub:** [Hugging Face](https://huggingface.co) (Qwen2.5-1.5B-Instruct
  base model)
- **Teacher model for data generation:** GROQ (`llama-3.3-70b-versatile`),
  used as an OpenAI-compatible drop-in replacement for OpenAI's API

## Project Steps (in order)

### 1. Setup
Install dependencies (transformers, datasets, LLaMA-Factory, json-repair,
faker, wandb), mount Google Drive for persistent storage of datasets and
trained models, and authenticate with Colab Secrets (`wandb`, `huggingface`,
`groq`).

### 2. Tasks (task definitions)
Define the two tasks the model must learn:
- A sample Arabic news story used for testing throughout the project.
- **Pydantic schemas** describing the exact JSON structure expected for each
  task (`NewsDetails` for extraction, `TranslatedStory` for translation).
  These schemas make the model's output verifiable — it's easy to check
  whether a required field is missing or a category is invalid.
- **Prompts** (system + user messages) that instruct the model how to
  perform each task and in what format to respond.

### 3. Baseline Evaluation
Load the *un-fine-tuned* base model and run it on the sample story for both
tasks. This is the reference point — without it, there's no way to measure
whether fine-tuning actually improved anything.

### 4. Evaluate with a Larger Model (Teacher)
Run the same tasks through a much larger model (via GROQ) to see the
target quality the small model should aim for.

### 5. Knowledge Distillation (data generation)
Fine-tuning requires thousands of labeled examples (story → correct JSON
output) — far too many to write by hand. The larger "teacher" model is used
to generate ideal outputs for a large batch of raw news stories (from
`news-sample.jsonl`), for both tasks. The results are saved to `sft.jsonl`.

*(In this project, `sft.jsonl` was pre-generated and reused rather than
regenerated, to save API costs and time.)*

### 6. Format Finetuning Datasets
Convert `sft.jsonl` into the `system / instruction / output` format that
LLaMA-Factory expects, shuffle the examples, and split them into:
- `train.json` (2,700 examples) — used to train the model
- `val.json` (66 examples) — held out, used to check the model generalizes
  rather than memorizes

### 7. Fine-tuning (LoRA training)
- Register the dataset paths in `dataset_info.json` so LLaMA-Factory can
  find them.
- Configure training in a YAML file: base model, LoRA settings, 3 epochs,
  batch size, learning rate, wandb logging, output directory.
- Run training via `llamafactory-cli train <config>.yaml`.
- The resulting **LoRA adapter** (a small set of extra weights, not a full
  copy of the model) is saved to Google Drive.

### 8. Fine-tuned Model Evaluation
Load the base model, attach the trained LoRA adapter with
`model.load_adapter(...)`, and re-run the same sample story through both
tasks. Compare the output against the baseline from step 3.

### 9. (Optional) vLLM & Cost Estimation
- **vLLM** — a high-throughput serving engine for deploying the fine-tuned
  model as an API.
- **Cost Estimation** — compares the token cost of running the small
  fine-tuned model vs. calling a commercial API like OpenAI for the same
  tasks at scale.

These two sections are optional and not required to understand or complete
the core fine-tuning workflow.

## Key Takeaways
- LoRA makes fine-tuning a small LLM on a free-tier GPU practical, since
  only a small fraction of parameters are updated.
- Structured output (JSON + schema) turns a generative task into something
  measurable — you can programmatically check correctness.
- Knowledge distillation (using a bigger model to generate training data
  for a smaller one) solves the "not enough labeled data" problem.
- A held-out validation set is what makes it possible to claim the model
  *generalized* rather than memorized the training examples.
