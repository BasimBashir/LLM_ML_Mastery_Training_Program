# LLM & Machine Learning Mastery Program

> **Start date:** 2026-10-05
> **Target pace:** 15–20 hours/week
> **Expected duration:** ~9–12 months at this pace
> **Primary hardware:** NVIDIA RTX 3090 24 GB VRAM, Intel Core i5-12600F, 64 GB DDR4 RAM
> **Operating principle:** Learn the theory → implement a minimal version → run a real experiment → evaluate → document.

---

## 0. Hardware strategy

Your RTX 3090 is excellent for **single-GPU LLM engineering, fine-tuning, inference, CUDA experiments, small-model pretraining, quantization, and RL/post-training experiments**.

It is **not** a practical machine for training modern 7B+ foundation models from scratch at production scale. The curriculum therefore deliberately uses:

- tiny models for algorithm implementation;
- ~100M–300M models for genuine pretraining experiments;
- 1B–3B models for practical SFT/DPO/GRPO/quantization;
- 7B–14B models primarily for inference and memory/optimization experiments;
- cloud/cluster hardware only when the learning objective genuinely requires multi-GPU scale.

### Hardware rules

- [ ] Install/verify NVIDIA driver and CUDA.
- [ ] Verify PyTorch sees the RTX 3090.
- [ ] Keep at least 100–150 GB SSD space free for datasets/checkpoints.
- [ ] Create a dedicated Python environment for the curriculum.
- [ ] Create a Git repository for all experiments.
- [ ] Create an experiment log containing: model, dataset, tokens, batch size, sequence length, LR, optimizer, GPU memory, throughput, loss, evaluation and conclusions.
- [ ] Never judge a training run only by loss; always maintain an evaluation set.

### Core references

- [PyTorch](https://pytorch.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)
- [Hugging Face TRL](https://huggingface.co/docs/trl/)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/)

---

# PHASE 0 — Environment & baseline

## Week 1 — Build your LLM laboratory

### Day 1 — Python/PyTorch environment

- [X] Create a clean Python environment.
- [X] Install PyTorch with CUDA support.
- [ ] Verify `torch.cuda.is_available()`.
- [ ] Verify RTX 3090 VRAM.
- [ ] Run a CUDA matrix-multiplication benchmark.
- [ ] Record GPU temperature, power and utilization during a benchmark.
- [ ] Install Git, Git LFS and Hugging Face CLI.
- [ ] Create the curriculum repository.

**Study**

- [PyTorch Learn](https://pytorch.org/learn/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

**Deliverable**

`00_environment/README.md`

Document your exact environment and benchmark.

### Day 2–3 — PyTorch training mechanics

- [ ] Implement a dataset.
- [ ] Implement a DataLoader.
- [ ] Implement a training loop without Trainer abstractions.
- [ ] Implement validation.
- [ ] Implement checkpoint saving/loading.
- [ ] Implement gradient accumulation.
- [ ] Implement mixed precision.
- [ ] Implement gradient clipping.
- [ ] Implement a learning-rate scheduler.

**Experiment**

Train a small MLP on MNIST/CIFAR-10 and intentionally cause:

- [ ] underfitting;
- [ ] overfitting;
- [ ] exploding gradients;
- [ ] bad learning-rate behavior.

### Day 4–7 — Deep-learning refresh

Review:

- [ ] SGD
- [ ] Adam
- [ ] AdamW
- [ ] cross entropy
- [ ] KL divergence
- [ ] softmax
- [ ] logits
- [ ] initialization
- [ ] normalization
- [ ] dropout
- [ ] weight decay
- [ ] warmup
- [ ] cosine LR decay
- [ ] gradient clipping
- [ ] BF16/FP16

**Study**

- [Dive into Deep Learning](https://d2l.ai/)
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/)

**Exit criterion**

You should be able to write a complete PyTorch training loop from memory.

---

# PHASE 1 — NLP foundations

## Week 2 — Language modeling before Transformers

- [ ] Bag-of-Words
- [ ] TF-IDF
- [ ] word embeddings
- [ ] Word2Vec
- [ ] CBOW
- [ ] Skip-gram
- [ ] negative sampling
- [ ] RNN
- [ ] LSTM
- [ ] GRU
- [ ] Seq2Seq
- [ ] teacher forcing
- [ ] encoder-decoder architecture

**Study**

- [Stanford NLP](https://web.stanford.edu/~jurafsky/slp3/)
- [CS224N](https://web.stanford.edu/class/cs224n/)
- [Dive into Deep Learning — NLP](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html)

**Implement**

- [ ] Word2Vec mini implementation.
- [ ] Tiny character-level RNN.
- [ ] Compare RNN vs LSTM training behavior.

**Deliverable**

`01_nlp_foundations/notes.md`

---

# PHASE 2 — Attention & Transformer

## Week 3 — Attention from first principles

- [ ] Query
- [ ] Key
- [ ] Value
- [ ] dot-product attention
- [ ] scaled dot-product attention
- [ ] masking
- [ ] causal masking
- [ ] multi-head attention
- [ ] self-attention
- [ ] cross-attention

Derive:

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k))V`

- [ ] Implement attention using raw PyTorch tensor operations.
- [ ] Visualize attention matrices.
- [ ] Implement causal masking yourself.

**Study**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy — Let&#39;s build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## Week 4 — Build GPT from scratch

- [ ] Token embeddings
- [ ] positional embeddings
- [ ] causal self-attention
- [ ] MLP block
- [ ] residual connections
- [ ] LayerNorm
- [ ] Transformer block
- [ ] LM head
- [ ] next-token loss
- [ ] autoregressive generation

**Major project #1**

### `mini-gpt-from-scratch`

Build a working GPT without using `transformers` for the model.

Requirements:

- [ ] train on a small corpus;
- [ ] generate text;
- [ ] save/load checkpoints;
- [ ] plot train/validation loss;
- [ ] calculate parameter count;
- [ ] calculate approximate training tokens;
- [ ] implement temperature;
- [ ] implement top-k sampling;
- [ ] implement top-p sampling.

**Study**

- [Andrej Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)
- [Andrej Karpathy minGPT](https://github.com/karpathy/minGPT)

**Exit criterion**

You can explain every tensor entering and leaving a Transformer block.

---

# PHASE 3 — Tokenization

## Week 5 — Tokenizers

- [ ] Character tokenization
- [ ] word tokenization
- [ ] BPE
- [ ] WordPiece
- [ ] Unigram
- [ ] SentencePiece
- [ ] vocabulary size
- [ ] special tokens
- [ ] BOS/EOS
- [ ] padding
- [ ] truncation
- [ ] chat templates
- [ ] token efficiency

**Implement**

- [ ] Build a tiny BPE tokenizer yourself.
- [ ] Train it on your own corpus.
- [ ] Compare it against a modern tokenizer.

**Study**

- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Let&#39;s build the GPT tokenizer — Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)

**Experiment**

Compare token counts for English, Urdu, Arabic, code and mixed technical text.

---

# PHASE 4 — Pretraining

## Weeks 6–10 — Foundation-model training

### Week 6 — Data

Learn:

- [ ] Common Crawl
- [ ] document extraction
- [ ] quality filtering
- [ ] language identification
- [ ] PII filtering
- [ ] spam filtering
- [ ] toxicity filtering
- [ ] exact deduplication
- [ ] near-duplicate detection
- [ ] MinHash
- [ ] data contamination
- [ ] train/validation leakage
- [ ] dataset mixtures

**Study**

- [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [Dolma](https://allenai.org/dolma)
- [The Pile](https://pile.eleuther.ai/)
- [RedPajama](https://www.together.ai/blog/redpajama)

### Week 7 — Pretraining mechanics

Learn:

- [ ] tokens/second
- [ ] FLOPs
- [ ] parameter count
- [ ] batch size
- [ ] microbatch
- [ ] gradient accumulation
- [ ] global batch size
- [ ] checkpointing
- [ ] validation loss
- [ ] perplexity
- [ ] training stability
- [ ] scaling laws

**Study**

- [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Stanford CS336 assignments](https://github.com/stanford-cs336)
- [Chinchilla](https://arxiv.org/abs/2203.15556)

### Week 8 — Train your first real language model

**Major project #2 — 50M–150M model**

- [ ] Build/prepare corpus.
- [ ] Train tokenizer.
- [ ] Create sharded dataset.
- [ ] Build GPT-style model.
- [ ] Train with BF16 where supported.
- [ ] Implement checkpoint recovery.
- [ ] Track loss and throughput.
- [ ] Evaluate perplexity.
- [ ] Generate samples.

### Week 9 — Scale the experiment

Train multiple configurations:

- [ ] ~50M model
- [ ] ~100M model
- [ ] ~150–300M model if feasible

Compare:

- [ ] parameters
- [ ] tokens
- [ ] compute
- [ ] training time
- [ ] validation loss
- [ ] throughput
- [ ] GPU memory

### Week 10 — Data ablations

- [ ] Train on unfiltered data.
- [ ] Train on filtered data.
- [ ] Train with deduplication.
- [ ] Compare domain mixtures.
- [ ] Measure downstream impact.

**Deliverable**

A technical report:

`pretraining_report.md`

---

# PHASE 5 — Modern Transformer architectures

## Weeks 11–13

### Week 11 — Modern GPT architecture

- [ ] RMSNorm
- [ ] Pre-Norm
- [ ] SwiGLU
- [ ] RoPE
- [ ] GQA
- [ ] MQA
- [ ] KV cache
- [ ] context length

**Study**

- [Llama 2 paper](https://arxiv.org/abs/2307.09288)
- [Llama 3 paper](https://arxiv.org/abs/2407.21783)
- [RoPE](https://arxiv.org/abs/2104.09864)

### Week 12 — Long context

- [ ] RoPE scaling
- [ ] sliding-window attention
- [ ] KV-cache memory
- [ ] chunked prefill
- [ ] long-context evaluation
- [ ] positional extrapolation

### Week 13 — Mixture of Experts

- [ ] sparse MoE
- [ ] router
- [ ] top-k routing
- [ ] expert capacity
- [ ] load balancing
- [ ] expert parallelism
- [ ] active vs total parameters

**Study**

- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Mixtral](https://arxiv.org/abs/2401.04088)
- [Megatron Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)

**Major project #3**

Implement a tiny MoE Transformer and compare it with a dense Transformer under approximately similar compute.

---

# PHASE 6 — SFT / Instruction Tuning

## Weeks 14–15

### Week 14 — SFT fundamentals

Learn:

- [ ] base vs instruct models
- [ ] instruction datasets
- [ ] conversation formatting
- [ ] chat templates
- [ ] loss masking
- [ ] packing
- [ ] truncation
- [ ] catastrophic forgetting
- [ ] synthetic instruction data

**Study**

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
- [Smol Course — Instruction Tuning](https://huggingface.co/learn/smol-course/unit1/1)
- [TRL documentation](https://huggingface.co/docs/trl/)

### Week 15 — SFT project

**Major project #4**

Fine-tune a 1B–3B model on a carefully constructed dataset.

- [ ] Build dataset.
- [ ] Split train/eval.
- [ ] Apply chat template.
- [ ] Train SFT.
- [ ] Evaluate base model.
- [ ] Evaluate SFT model.
- [ ] Create qualitative evaluation set.
- [ ] Analyze failures.
- [ ] Publish training configuration.

---

# PHASE 7 — PEFT / LoRA / QLoRA

## Weeks 16–17

Learn:

- [ ] full fine-tuning
- [ ] LoRA mathematics
- [ ] rank
- [ ] alpha
- [ ] target modules
- [ ] dropout
- [ ] adapter merging
- [ ] QLoRA
- [ ] NF4
- [ ] double quantization
- [ ] paged optimizers

**Study**

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/)

**Major project #5**

Run:

- [ ] full/partial fine-tuning where feasible;
- [ ] LoRA;
- [ ] QLoRA.

Record:

- [ ] VRAM
- [ ] trainable parameters
- [ ] training time
- [ ] checkpoint size
- [ ] evaluation quality

---

# PHASE 8 — Evaluation

## Week 18

Learn:

- [ ] perplexity
- [ ] exact match
- [ ] F1
- [ ] MMLU
- [ ] MMLU-Pro
- [ ] GSM8K
- [ ] GPQA
- [ ] HumanEval
- [ ] MBPP
- [ ] IFEval
- [ ] LongBench
- [ ] LLM-as-judge
- [ ] pairwise evaluation
- [ ] judge bias
- [ ] contamination

**Study**

- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/)
- [OpenAI Evals](https://github.com/openai/evals)

**Major project #6**

Build your own evaluation harness:

```text
model
 ↓
prompt set
 ↓
generation
 ↓
scoring
 ↓
metrics
 ↓
comparison report
```

---

# PHASE 9 — Preference Learning

## Weeks 19–21

### Week 19 — RLHF theory

Learn:

- [ ] reward
- [ ] policy
- [ ] value function
- [ ] advantage
- [ ] policy gradient
- [ ] REINFORCE
- [ ] actor-critic
- [ ] PPO
- [ ] KL regularization
- [ ] reward model

**Study**

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [TRL RLHF resources](https://huggingface.co/docs/trl/)

### Week 20 — DPO

Learn:

- [ ] preference datasets
- [ ] chosen/rejected responses
- [ ] reference model
- [ ] DPO objective
- [ ] beta
- [ ] reward margin
- [ ] preference probability

**Study**

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)

**Major project #7**

Take your SFT model:

```text
SFT
 ↓
preference dataset
 ↓
DPO
 ↓
evaluation
```

Compare SFT vs DPO.

### Week 21 — DPO family

Learn:

- [ ] IPO
- [ ] KTO
- [ ] ORPO
- [ ] SimPO
- [ ] CPO
- [ ] online vs offline preference optimization

**Study**

- [ORPO](https://arxiv.org/abs/2403.07691)
- [TRL trainers](https://huggingface.co/docs/trl/)

**Experiment**

Train at least two preference-optimization methods against the same baseline.

---

# PHASE 10 — RL for LLMs & reasoning

## Weeks 22–25

### Week 22 — RL fundamentals

- [ ] MDP
- [ ] state
- [ ] action
- [ ] reward
- [ ] return
- [ ] policy
- [ ] value
- [ ] advantage
- [ ] policy gradient
- [ ] PPO

**Study**

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/)

### Week 23 — Reward modeling

- [ ] pairwise preference model
- [ ] Bradley-Terry model
- [ ] reward normalization
- [ ] reward hacking
- [ ] reward overoptimization
- [ ] reward model evaluation

### Week 24 — GRPO

Learn:

- [ ] group sampling
- [ ] relative rewards
- [ ] advantages
- [ ] KL control
- [ ] verifier-based reward

**Study**

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)

### Week 25 — RLVR

Build:

```text
problem
 ↓
N model solutions
 ↓
automatic verifier
 ↓
reward
 ↓
GRPO
 ↓
new model
```

Use math/code problems where correctness can be automatically verified.

**Major project #8 — Tiny Reasoning Model**

- [ ] Generate multiple solutions.
- [ ] Build verifier.
- [ ] Train with GRPO/RLVR.
- [ ] Compare against SFT/DPO baseline.
- [ ] Analyze reward hacking.
- [ ] Evaluate on held-out problems.

---

# PHASE 11 — Distillation & Synthetic Data

## Weeks 26–27

Learn:

- [ ] knowledge distillation
- [ ] logits distillation
- [ ] response distillation
- [ ] synthetic instruction data
- [ ] rejection sampling
- [ ] self-instruct
- [ ] teacher/student training
- [ ] synthetic preference data

**Study**

- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [TRL distillation](https://huggingface.co/docs/trl/)

**Project**

```text
large teacher
 ↓
synthetic dataset
 ↓
small student
 ↓
SFT/distillation
 ↓
evaluation
```

---

# PHASE 12 — Inference Engineering

## Weeks 28–30

### Week 28 — Memory

Understand:

- [ ] weights
- [ ] gradients
- [ ] optimizer states
- [ ] activations
- [ ] KV cache
- [ ] BF16
- [ ] FP16
- [ ] FP8
- [ ] INT8
- [ ] INT4

Calculate model memory manually.

### Week 29 — Quantization

Learn:

- [ ] GPTQ
- [ ] AWQ
- [ ] bitsandbytes
- [ ] GGUF
- [ ] weight-only quantization
- [ ] activation quantization
- [ ] KV-cache quantization

**Study**

- [vLLM Quantization](https://docs.vllm.ai/en/stable/features/quantization/)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Week 30 — Serving

Learn:

- [ ] prefill
- [ ] decode
- [ ] KV cache
- [ ] continuous batching
- [ ] PagedAttention
- [ ] prefix caching
- [ ] speculative decoding
- [ ] TTFT
- [ ] inter-token latency
- [ ] throughput

**Study**

- [vLLM](https://docs.vllm.ai/)
- [SGLang](https://docs.sglang.ai/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

**Major project #9**

Turn a model into an OpenAI-compatible local API.

Benchmark:

- [ ] latency
- [ ] TTFT
- [ ] tokens/sec
- [ ] concurrent requests
- [ ] VRAM
- [ ] quantization quality

---

# PHASE 13 — Distributed Training

## Weeks 31–34

Your single RTX 3090 is enough to **learn the concepts and run toy distributed experiments**, but real multi-GPU scaling requires additional hardware.

### Week 31

- [ ] multiprocessing
- [ ] NCCL
- [ ] all-reduce
- [ ] all-gather
- [ ] reduce-scatter
- [ ] DDP

**Study**

- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Week 32

- [ ] FSDP
- [ ] parameter sharding
- [ ] gradient sharding
- [ ] optimizer sharding
- [ ] activation checkpointing

**Study**

- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)

### Week 33

- [ ] ZeRO-1
- [ ] ZeRO-2
- [ ] ZeRO-3
- [ ] DeepSpeed

**Study**

- [DeepSpeed](https://www.deepspeed.ai/)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)

### Week 34

- [ ] tensor parallelism
- [ ] pipeline parallelism
- [ ] sequence/context parallelism
- [ ] expert parallelism

- [3D parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

**Study**

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [Megatron Core QuickStart](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/QuickStart.md)

**Major project #10**

Run a toy Transformer with distributed training concepts. If you later get a second GPU, repeat with real multi-GPU TP/PP experiments.

---

# PHASE 14 — GPU & Kernel Engineering

## Weeks 35–37

### Week 35 — CUDA fundamentals

- [ ] threads
- [ ] blocks
- [ ] warps
- [ ] shared memory
- [ ] global memory
- [ ] registers
- [ ] occupancy
- [ ] memory bandwidth
- [ ] compute throughput

**Study**

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

### Week 36 — FlashAttention

- [ ] understand IO complexity
- [ ] memory hierarchy
- [ ] tiling
- [ ] fused attention

**Study**

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)

### Week 37 — Triton

- [ ] write a Triton kernel
- [ ] benchmark against PyTorch
- [ ] implement vector addition
- [ ] implement softmax
- [ ] implement matmul
- [ ] profile kernels

**Study**

- [OpenAI Triton](https://github.com/triton-lang/triton)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

---

# PHASE 15 — RAG at engineering depth

## Weeks 38–39

Learn:

- [ ] BM25
- [ ] dense retrieval
- [ ] hybrid retrieval
- [ ] embeddings
- [ ] cross-encoder reranking
- [ ] query expansion
- [ ] HyDE
- [ ] multi-query retrieval
- [ ] contextual retrieval
- [ ] Graph RAG
- [ ] hierarchical retrieval
- [ ] retrieval evaluation

Metrics:

- [ ] Recall@K
- [ ] Precision@K
- [ ] MRR
- [ ] nDCG
- [ ] answer relevance
- [ ] faithfulness

**Study**

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Qdrant](https://qdrant.tech/documentation/)
- [BEIR](https://github.com/beir-cellar/beir)

**Major project #11**

Build an evaluation-driven RAG system.

Do not just measure answer quality. Measure retrieval quality separately.

---

# PHASE 16 — Agents

## Weeks 40–42

### Learn

- [ ] function calling
- [ ] tool schemas
- [ ] ReAct
- [ ] planning
- [ ] memory
- [ ] tool execution
- [ ] browser agents
- [ ] coding agents
- [ ] multi-agent systems
- [ ] agent evaluation
- [ ] observability

**Study**

- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

**Major project #12**

Build an agent that:

```text
LLM
 ↓
planner
 ↓
tool selection
 ↓
tool execution
 ↓
observation
 ↓
next action
```

Add:

- [ ] tool failure recovery
- [ ] trace logging
- [ ] evaluation set
- [ ] cost tracking
- [ ] latency tracking

---

# PHASE 17 — Multimodal LLMs

## Weeks 43–46

### Vision

- [ ] ViT
- [ ] CLIP
- [ ] image encoder
- [ ] projector
- [ ] multimodal tokens
- [ ] VLM SFT
- [ ] visual reasoning

**Study**

- [CLIP](https://arxiv.org/abs/2103.00020)
- [ViT](https://arxiv.org/abs/2010.11929)
- [Hugging Face Vision-Language Models](https://huggingface.co/docs/transformers/tasks/image_text_to_text)

### Audio

- [ ] speech encoders
- [ ] Whisper
- [ ] speech tokens
- [ ] audio-language models
- [ ] speech-to-speech

**Study**

- [Whisper](https://github.com/openai/whisper)

### Video

- [ ] frame sampling
- [ ] temporal embeddings
- [ ] video tokens
- [ ] multimodal context

**Major project #13**

Build or fine-tune a small VLM and evaluate it on a custom image QA dataset.

---

# PHASE 18 — Interpretability & Safety

## Weeks 47–49

### Interpretability

- [ ] activation analysis
- [ ] probing
- [ ] causal tracing
- [ ] activation patching
- [ ] sparse autoencoders
- [ ] feature discovery
- [ ] mechanistic interpretability

**Study**

- [Transformer Circuits](https://transformer-circuits.pub/)
- [Neel Nanda — Interpretability](https://www.neelnanda.io/)
- [ARENA](https://www.arena.education/)

### Safety/alignment

- [ ] hallucination
- [ ] sycophancy
- [ ] jailbreaks
- [ ] reward hacking
- [ ] specification gaming
- [ ] adversarial evaluation
- [ ] red teaming
- [ ] RLHF
- [ ] RLAIF
- [ ] Constitutional AI

**Study**

- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Anthropic Research](https://www.anthropic.com/research)

---

# PHASE 19 — Research mode

## Weeks 50+

At this point stop treating courses as your primary source.

Your workflow becomes:

```text
paper
 ↓
understand
 ↓
derive
 ↓
implement
 ↓
reproduce
 ↓
benchmark
 ↓
find discrepancy
 ↓
hypothesis
 ↓
experiment
 ↓
paper/report
```

### Build a paper-reading system

For every paper:

- [ ] Read abstract.
- [ ] Identify the problem.
- [ ] Identify the claimed contribution.
- [ ] Read architecture/method.
- [ ] Derive important equations.
- [ ] Read experiments.
- [ ] Identify baselines.
- [ ] Identify limitations.
- [ ] Reproduce the smallest meaningful experiment.
- [ ] Write your own critique.

### Paper sequence

- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ] [GPT-3](https://arxiv.org/abs/2005.14165)
- [ ] [Scaling Laws](https://arxiv.org/abs/2001.08361)
- [ ] [Chinchilla](https://arxiv.org/abs/2203.15556)
- [ ] [Llama 2](https://arxiv.org/abs/2307.09288)
- [ ] [InstructGPT](https://arxiv.org/abs/2203.02155)
- [ ] [LoRA](https://arxiv.org/abs/2106.09685)
- [ ] [QLoRA](https://arxiv.org/abs/2305.14314)
- [ ] [DPO](https://arxiv.org/abs/2305.18290)
- [ ] [FlashAttention](https://arxiv.org/abs/2205.14135)
- [ ] [ZeRO](https://arxiv.org/abs/1910.02054)
- [ ] [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [ ] [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [ ] [ORPO](https://arxiv.org/abs/2403.07691)
- [ ] [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

Then continuously select current model technical reports and recent papers from:

- [ ] [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent)
- [ ] [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent)
- [ ] [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent)
- [ ] [Hugging Face Papers](https://huggingface.co/papers)
- [ ] [OpenReview](https://openreview.net/)

---

# CAPSTONE — Build your own complete mini foundation-model stack

Do this only after completing the core phases.

## Capstone architecture

```text
RAW DATA
   ↓
DATA PIPELINE
   ↓
DEDUP / QUALITY FILTER
   ↓
TOKENIZER
   ↓
PRETRAINING
   ↓
BASE MODEL
   ↓
SFT
   ↓
PREFERENCE DATA
   ↓
DPO
   ↓
RLVR / GRPO
   ↓
EVALUATION
   ↓
QUANTIZATION
   ↓
vLLM / llama.cpp
   ↓
RAG
   ↓
TOOLS / AGENT
   ↓
MONITORING
```

### Capstone checklist

- [ ] Own dataset pipeline.
- [ ] Own tokenizer.
- [ ] Own Transformer implementation.
- [ ] Pretrained model.
- [ ] SFT model.
- [ ] Preference dataset.
- [ ] DPO model.
- [ ] RL/GRPO experiment.
- [ ] Evaluation suite.
- [ ] Quantized model.
- [ ] Production-style inference server.
- [ ] RAG layer.
- [ ] Tool-calling layer.
- [ ] Agent.
- [ ] Benchmark report.
- [ ] Technical write-up.
- [ ] GitHub repository.
- [ ] Model card.

---

# Daily operating schedule

For a normal 15–20 hour week:

### Monday — Theory

- [ ] 2–3 hours lecture/book/paper
- [ ] 1 hour mathematical derivation

### Tuesday — Implementation

- [ ] 3–4 hours implementation

### Wednesday — Implementation

- [ ] 3–4 hours implementation

### Thursday — Experiment

- [ ] 2–3 hours training/benchmarking
- [ ] 1 hour experiment analysis

### Friday — Paper/research

- [ ] 2–3 hours paper reading

### Saturday — Deep work

- [ ] 3–5 hours project work

### Sunday

- [ ] Review completed checkboxes.
- [ ] Write experiment notes.
- [ ] Record failures.
- [ ] Plan the next week.
- [ ] Rest if the weekly objectives are complete.

---

# Definition of "COMPLETE"

Do not check a topic merely because you watched a lecture.

A topic is complete only when:

- [ ] I can explain it without notes.
- [ ] I understand why it exists.
- [ ] I understand the mathematics at the required level.
- [ ] I implemented a minimal version.
- [ ] I used the production implementation.
- [ ] I ran an experiment.
- [ ] I measured the result.
- [ ] I documented what happened.
- [ ] I understand at least one failure mode.

---

# Your first 7 days — START HERE

Do **not** jump to DPO, RLHF or agents yet.

## Day 1

- [ ] Set up Python/PyTorch/CUDA.
- [ ] Verify RTX 3090.
- [ ] Create Git repository.
- [ ] Create experiment log.
- [ ] Read PyTorch training-loop material.

## Day 2

- [ ] Review tensors/autograd.
- [ ] Implement an MLP training loop manually.
- [ ] Add checkpointing.

## Day 3

- [ ] Implement AdamW.
- [ ] Add gradient accumulation.
- [ ] Add mixed precision.
- [ ] Add LR scheduling.

## Day 4

- [ ] Learn embeddings.
- [ ] Learn language modeling.
- [ ] Implement a bigram language model.

## Day 5

- [ ] Implement character-level language modeling.
- [ ] Plot training/validation loss.

## Day 6

- [ ] Watch/read Karpathy's GPT-from-scratch material.
- [ ] Implement single-head attention.

## Day 7

- [ ] Implement multi-head causal attention.
- [ ] Write `week01_report.md`.
- [ ] Record what you understand and what remains unclear.

### Week 1 exit test

You should be able to answer, without searching:

1. What exactly is a logit?
2. Why does softmax appear in language modeling?
3. Why is cross entropy used?
4. What does AdamW do?
5. Why do gradients accumulate?
6. What does BF16 change?
7. What is an embedding?
8. What is next-token prediction?
9. Why must GPT use causal masking?
10. What are Q, K and V?

If you cannot answer these, **do not advance**.

---

# Master progress tracker

## Foundations

- [ ] PyTorch training
- [ ] Math refresh
- [ ] NLP foundations
- [ ] Attention
- [ ] Transformer
- [ ] Tokenization

## Foundation-model training

- [ ] Data engineering
- [ ] Pretraining
- [ ] Scaling laws
- [ ] Modern architectures
- [ ] RoPE
- [ ] GQA/MQA
- [ ] MoE
- [ ] Long context

## Post-training

- [ ] SFT
- [ ] LoRA
- [ ] QLoRA
- [ ] DPO
- [ ] IPO
- [ ] KTO
- [ ] ORPO
- [ ] Reward modeling
- [ ] RLHF
- [ ] PPO
- [ ] GRPO
- [ ] RLVR
- [ ] Distillation
- [ ] Synthetic data

## Systems

- [ ] Quantization
- [ ] KV cache
- [ ] vLLM
- [ ] SGLang
- [ ] CUDA
- [ ] Triton
- [ ] FlashAttention
- [ ] DDP
- [ ] FSDP
- [ ] ZeRO
- [ ] Tensor Parallelism
- [ ] Pipeline Parallelism
- [ ] Context Parallelism
- [ ] Expert Parallelism
- [ ] Megatron

## Applications

- [ ] RAG
- [ ] Retrieval evaluation
- [ ] Reranking
- [ ] Agents
- [ ] Tool calling
- [ ] Multimodal
- [ ] Vision-language
- [ ] Audio-language
- [ ] Video-language

## Research

- [ ] Interpretability
- [ ] Mechanistic interpretability
- [ ] Safety
- [ ] Alignment
- [ ] Paper reproduction
- [ ] Independent research project
- [ ] Technical report
- [ ] Open-source release

---

# Final target

When every major section is complete, you should be able to independently answer:

> **How would I build a language model from raw data, train it, instruction-tune it, align it, teach it to reason, evaluate it, optimize it, distribute its training, serve it efficiently, connect it to tools/RAG, and investigate its internal behavior?**

If you can implement the major pieces and defend the engineering decisions quantitatively, you have moved beyond "learning LLMs" into serious **LLM/ML engineering and research**.

---

## Recommended primary resources

1. [Stanford CS336](https://cs336.stanford.edu/) — primary from-scratch curriculum.
2. [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/) — practical ecosystem.
3. [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/) — post-training.
4. [Hugging Face TRL](https://huggingface.co/docs/trl/) — SFT/DPO/RL/distillation.
5. [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) — implementation intuition.
6. [PyTorch](https://pytorch.org/) — core framework.
7. [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/) — large-scale training.
8. [vLLM](https://docs.vllm.ai/) — production inference.
9. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — evaluation.
10. [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/) — agentic systems.
11. [Triton](https://triton-lang.org/) — GPU kernels.
12. [FlashAttention](https://github.com/Dao-AILab/flash-attention) — attention optimization.
13. [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) — current NLP research.
14. [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) — current ML research.
15. [Hugging Face Papers](https://huggingface.co/papers) — practical paper discovery.
