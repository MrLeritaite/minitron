# minitron

A from-scratch, 4D-parallel LLM training framework, built to deeply understand
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Nanotron](https://github.com/huggingface/nanotron).

**Goal (Nov 16 → Dec 31, 2026):** every component written by hand and verified against a
single-device reference: DP + ZeRO-1/2/3 → TP + SP → PP (AFAB, 1F1B, interleaved) → CP (ring attention) → ⭐ MoE/EP.
**January extension:** CUDA/Triton kernels, FlashAttention in Triton, a minimal comm layer.

> Golden rule: no parallel feature is "done" until its loss and gradients match the single-device reference.

## Layout

```
minitron/   the framework (written from scratch, week by week)
scripts/    entrypoints (training, benchmarks, hello_dist.py)
tests/      correctness tests vs the single-device reference
notes/      daily notes, derivations, weekly reviews
refs/       reference repos (git-ignored, read AFTER your own attempt)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/run.sh 4 scripts/hello_dist.py   # day 1: 4 ranks on CPU (gloo)
```

`scripts/run.sh` wraps `torchrun` with loopback settings — on macOS, `torchrun --standalone`
hangs trying to resolve the IPv6 hostname, so it pins `--master_addr=127.0.0.1` and
`GLOO_SOCKET_IFNAME=lo0`. On a Linux GPU box plain `torchrun` works and NCCL is picked automatically.

Reference repos (cloned into `refs/`, not committed):

```bash
cd refs
git clone https://github.com/huggingface/picotron
git clone https://github.com/huggingface/picotron_tutorial
git clone https://github.com/huggingface/nanotron
git clone https://github.com/NVIDIA/Megatron-LM
git clone https://github.com/karpathy/nanoGPT
```

## Roadmap

- [ ] **W1** (Nov 16–22): reference GPT model, memory calculator, hand-written ring all-reduce
- [ ] **W2** (Nov 23–29): DDP (bucketed + overlapped), ZeRO-1/2/3, bf16 mixed precision
- [ ] **W3** (Nov 30–Dec 6): tensor parallelism, vocab-parallel CE, sequence parallelism, recompute
- [ ] **W4** (Dec 7–13): pipeline parallelism (AFAB, 1F1B, interleaved), 3D grid, first GPU profiling
- [ ] **W5** (Dec 14–20): context parallelism (ring attention), sharded checkpoints, ⭐ MoE + EP
- [ ] **W6** (Dec 21–27): read Nanotron + Megatron-Core end to end; mapping doc
- [ ] **W7** (Dec 28–31): capstone on 8×H100 vs Nanotron, write-up
- [ ] **W8–W11** (Jan 2027): CUDA → Triton kernels → FlashAttention → custom comm layer

## Key reading

- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- Megatron-LM ([1909.08053](https://arxiv.org/abs/1909.08053)), PTD-P ([2104.04473](https://arxiv.org/abs/2104.04473)),
  Sequence parallelism ([2205.05198](https://arxiv.org/abs/2205.05198)), ZeRO ([1910.02054](https://arxiv.org/abs/1910.02054))
