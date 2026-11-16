# Week 1 — Foundations (Nov 16–22)

## Day 1 · Mon Nov 16 — Setup
- [x] Repo, venv (Python 3.12 + torch), reference repos in `refs/`
- [x] `scripts/run.sh 4 scripts/hello_dist.py` → 4 ranks print + all_reduce OK
- [ ] Read: Ultra-Scale Playbook — intro + "First steps: training on one GPU"
- [ ] Think: what dominates memory and time in training?

Notes:

## Day 2 · Tue Nov 17 — Memory & FLOPs
- [ ] `scripts/memory_calc.py`: params / grads / optimizer / activations for (L, h, s, b, V)
- [ ] Derive: 16 bytes/param for mixed-precision Adam; 6·N·tokens FLOPs; MFU

Notes:
