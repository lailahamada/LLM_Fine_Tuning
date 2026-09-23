# vLLM: What It Is, How It Works, and When to Use It

## What is vLLM?

**vLLM** is an open-source library for serving (running inference on) LLMs.
Its goal is to let a model handle a **large number of concurrent requests**
with maximum throughput and minimal GPU memory waste. The difference from
calling `model.generate()` directly (e.g. via `transformers`) is that vLLM
is built for production serving — a model answering thousands of users —
not for a single request in a notebook.

## How It Works Technically

**The problem it solves:** when a model generates text, it keeps a temporary
memory buffer called the **KV Cache** for every request (all the tokens
generated so far in that conversation). The traditional approach reserves a
large, **fixed** block of memory per request up front (assuming the longest
possible response), even if the actual response is short. This wastes a lot
of GPU memory and limits how many requests can be served at once.

**vLLM's solution relies on two core techniques:**

1. **PagedAttention** — instead of reserving one large contiguous memory
   block per request, the KV Cache is split into small pages (similar to
   virtual memory in operating systems), and memory is allocated
   incrementally, only as actually needed. Result: far less wasted memory,
   so the same GPU can serve many more requests.

2. **Continuous Batching** — instead of waiting for every request in a
   batch to finish before starting a new batch (the traditional approach),
   vLLM adds new requests into the batch as soon as a slot frees up (a
   request finishes), without waiting for the rest. This significantly
   increases GPU utilization.

**Practical result:** vLLM achieves much higher throughput (often reported
as 2–24x, depending on the scenario) compared to plain
`transformers.generate()`, especially when many requests arrive at the same
time.

## How to Benefit From It

You run it as an **API server** instead of calling the model directly in
code:

```bash
vllm serve /gdrive/MyDrive/llm-finetuning/models --port 8000
```

Then anyone (or any code) can send it requests just like the OpenAI API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="qwen-news-analyzer",
    messages=[...]
)
```

## Do We Actually Need It Here?

**No, not at this stage.** vLLM solves the problem of **serving under load**:
thousands of users sending requests at the same moment, all needing fast
responses instead of waiting in line one by one.

In this project, we only do two things:
- Train the model (LoRA fine-tuning).
- Test it on **one story at a time**, to confirm the output improved after
  training.

There's no "concurrent requests" or "many users" here, so plain
`model.generate()` is entirely sufficient — vLLM would just be unnecessary
overhead with no real benefit.

## A Real-World Case Where You WOULD Need It

Imagine that after finishing training, you build a **website or app** that
receives news from real users throughout the day — a news site, for
example, that sends every newly published article to your server to
automatically extract details and translate it. If 200 articles arrive in
the same minute during peak time, and each one needs fast processing:

- **Without vLLM:** the server processes requests sequentially or with
  naive batching, and response time degrades sharply as the number of
  requests grows — you'd need more GPUs just to keep up with the same load.
- **With vLLM:** the same GPU can serve far more concurrent requests with
  lower latency, because memory is managed efficiently and batching is
  continuous.

**In short: vLLM matters at deployment time, when a model serves real
traffic — not during training or experimentation.** That's why it's the
last, optional section in the notebook, after training and evaluation are
done.

---

## How vLLM Relates to Load Balancing and Model Routing

These three concepts sound similar but solve **different problems**, at
**different levels** of a production system.

### Continuous Batching (what vLLM does)
Happens **inside a single model instance, on a single GPU, at the same
moment**. Different requests are grouped into one "batch" processed
together on the GPU, and as soon as one request finishes, a new one takes
its slot immediately instead of waiting. This is **scheduling within one
model** — there's no more than one copy of the model involved; that one
copy just manages its time across requests intelligently.

### Load Balancing
Happens **above the model level, across multiple servers/instances**.
Imagine the same model is duplicated across 5 different GPUs (5 vLLM
instances running in parallel). The Load Balancer is the component that
receives all incoming user requests and decides **which of the 5 instances**
handles each one, based on who's free or under less load.

```
Users → Load Balancer → [vLLM instance 1]
                       → [vLLM instance 2]
                       → [vLLM instance 3]
```

**Key difference from continuous batching:** continuous batching improves
efficiency **inside** one model copy. Load balancing distributes load
**across** multiple copies. Both can be used together: each of the 5
instances has continuous batching running inside it, and the load balancer
distributes traffic between them.

### Model Routing
This is different from both of the above. Model routing applies when you
have **different models** (not copies of the same model), and need to
decide **which model best fits each request**.

Example: you have a small, fast, cheap model (like your fine-tuned
Qwen 1.5B) and a larger, more powerful, more expensive one (like gpt-4o).
The Router looks at the incoming request and decides:
- A simple, routine task → send it to the small, cheap model.
- A complex task requiring deep reasoning → send it to the large model.

```
Users → Router → [Small fast model] (simple query)
               → [Large expensive model] (complex query)
```

The routing criterion could be task type, required confidence level, cost,
or even the output of a classifier that estimates the request's
"difficulty."

### Summary Table

| Concept | Happens where | Solves what | Example |
|---|---|---|---|
| **Continuous Batching** | Inside one model instance | Better use of GPU memory/time | vLLM efficiently managing multiple requests on one GPU |
| **Load Balancing** | Across multiple copies of the **same** model | Distributing load when one instance isn't enough | 5 Qwen servers, traffic split between them |
| **Model Routing** | Across **different** models | Picking the right model per task (cost/quality trade-off) | Simple question → small model, complex question → large model |

### Are They Related?

Yes — in a real large-scale production system, they complement each other:
a **Router** might first decide which model is appropriate, then, once it
reaches a specific model, a **Load Balancer** distributes the request
across that model's multiple instances, and inside each instance,
**continuous batching** manages requests efficiently. Each one solves a
completely different problem, and none of them substitutes for another.

In this project's current stage — one instance of one model, tested on one
story at a time — none of these three is actually needed. They all become
relevant once you move to **real production deployment at scale**.
