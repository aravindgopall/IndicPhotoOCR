# GPU Runbook — Chitrapathak-2 LoRA SFT for Marathi OCR

One-shot recipe to fine-tune Chitrapathak-2 (3B) on our real Marathi data and
evaluate against the zero-shot baseline. Everything was smoke-tested locally
(train → adapter → merge → eval) on 2026-09-08.

## What you're producing

- **LoRA adapter** trained on 9,926 samples (8,886 real fragments +
  130 real lines ×8) — no synthetic in round 1 (style comes from real data)
- **Merged model** + eval numbers on the 3 fixed held-out benchmarks

Baselines to beat (zero-shot, from `data/marathi_vlm_pivot_results.txt`):
| Benchmark | Zero-shot CP2 | D (PARSeq prod) |
|---|---|---|
| 90 lines characc | **96.01** | 87.0–88.5 |
| 90 lines exact | 25.6% | 5.6% |
| 302 clean crops exact | 66.89% | 83.44% |
| 75 short crops exact | 52.00% | 73.33% |

Goal of SFT: kill the space-normalization habit (~83% of line failures:
`क्र.पदनि` → `क्र. पदनि`) → line characc ~97-98%, exact 40-60%, WITHOUT
losing fragment-crop accuracy (we train on fragments too).

## 1. Rent

- **RunPod** (or Lambda/Vast): **RTX 4090 24GB** (~$0.45/hr) is plenty;
  A100 40GB (~$1.7/hr) if you want speed. Expected: **1-2h train + ~25min
  eval → under $3 total**.
- Image: a **PyTorch >= 2.5 + CUDA 12.x** pod (RunPod's "RunPod Pytorch 2.x"
  template — check `python -c "import torch; print(torch.__version__)"`;
  transformers 5.x silently disables torch on older builds, see step 3).
  Disk: 60GB.

## 2. Upload the package (from this Mac)

```bash
scp /tmp/cp2_sft_package.tar.gz root@<POD_IP>:/workspace/
```

(67MB: scripts + train data + held-out evals. The 7GB base model downloads
from HuggingFace on the pod — do NOT upload it.)

## 3. Setup on the pod

```bash
cd /workspace && tar -xzf cp2_sft_package.tar.gz
# transformers 5.x requires torch >= 2.5. Pods shipping 2.4.x fail with
# "AutoModelForImageTextToText requires the PyTorch library but it was not found".
python -c "import torch; print(torch.__version__)"
# if < 2.5 (CUDA 12.4 wheels; use cu126 on a CUDA 12.6+ pod):
# upgrade torchvision/torchaudio in the SAME command — a stale torchaudio built
# against the old torch dies with "undefined symbol: _ZN2at4_ops..." on import.
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

pip install -q "transformers==5.16.1" peft accelerate
# sanity: must print True
python -c "import transformers, torch; print(torch.__version__, transformers.utils.is_torch_available())"
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download('krutrim-ai-labs/Chitrapathak-2', local_dir='/workspace/data/chitrapathak2')
print('model at', p)
EOF
```

## 4. Train (~1-2h on 4090)

```bash
cd /workspace
python scripts/sft_chitrapathak.py \
    --model data/chitrapathak2 \
    --data data/cp2_sft/train.jsonl \
    --out out/cp2_sft_lora \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 \
    --device cuda
```

Notes:
- LoRA r=32/alpha=64 on the 252 language-model projections (vision tower
  frozen). Trainable: 59.9M params (1.57%).
- Loss is masked to the assistant response; the prompt/image tokens are
  excluded (matches inference exactly).
- Same prompt + same max_pixels=300k as the zero-shot eval — no train/eval
  distribution shift.
- Watch the printed `loss` (starts ~0.7, should fall < 0.15) and `eta`.
- If OOM: drop `--batch-size` to 2 (keep grad-accum 4 → effective 8).

## 4b. Run it unattended (so you can close the laptop)

`nohup` survives SSH disconnect — no tmux needed (RunPod images often lack it).
Chain train → merge → eval into one script so the whole run finishes without you:

```bash
cd /workspace && mkdir -p logs
cat > run_all.sh <<'EOF'
#!/bin/bash
set -euo pipefail
cd /workspace

python -u scripts/sft_chitrapathak.py \
    --model data/chitrapathak2 --data data/cp2_sft/train.jsonl \
    --out out/cp2_sft_lora \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 --device cuda

python -u scripts/sft_chitrapathak.py --merge-only \
    --model data/chitrapathak2 --adapter out/cp2_sft_lora \
    --out out/cp2_sft_merged

python -u scripts/eval_chitrapathak.py \
    --test-jsonl data/marathi_v2_test_lines/labels.jsonl \
    --test-dir data/marathi_v2_test_lines --model-path out/cp2_sft_merged \
    --max-new-tokens 320 --max-pixels 300000 --out out/preds_lines.jsonl

python -u scripts/eval_chitrapathak.py \
    --test-jsonl data/marathi_v2_test_det_clean_fixed/labels.jsonl \
    --test-dir data/marathi_v2_test_det_clean --model-path out/cp2_sft_merged \
    --out out/preds_clean.jsonl

python -u scripts/eval_chitrapathak.py \
    --test-jsonl data/marathi_v3_short_test_det_fixed/labels.jsonl \
    --test-dir data/marathi_v3_short_test_det --model-path out/cp2_sft_merged \
    --out out/preds_short.jsonl

echo "ALL DONE $(date -u)"
EOF
chmod +x run_all.sh

nohup ./run_all.sh > logs/run.log 2>&1 &
echo $! > logs/run.pid
disown
```

Now log off. To check back in:

```bash
tail -f logs/run.log                     # live progress (Ctrl-C just stops tailing)
tail -5 logs/run.log; nvidia-smi         # quick status
ps -p "$(cat logs/run.pid)" >/dev/null && echo RUNNING || echo "FINISHED/DIED"
grep -c "ALL DONE" logs/run.log          # 1 = pipeline completed
kill "$(cat logs/run.pid)"               # abort
```

`python -u` is required — without it stdout is block-buffered into the log file
and `tail -f` shows nothing for minutes.

**`nohup` survives SSH disconnect, not pod shutdown.** Leave the pod *running*
(closing the browser/terminal is fine); "Stop pod" kills the process and, on
RunPod, wipes anything outside `/workspace`. Note the trainer only writes the
adapter **once, after the final step** (no per-epoch checkpoints, no resume) —
a run killed at 90% has to start over.

## 5. Merge (~3 min)

```bash
python scripts/sft_chitrapathak.py --merge-only \
    --model data/chitrapathak2 --adapter out/cp2_sft_lora \
    --out out/cp2_sft_merged
```

## 6. Evaluate (~25 min on GPU) — the three fixed benchmarks

```bash
# LINES (headline metric: characc vs 96.01 zero-shot)
python scripts/eval_chitrapathak.py \
    --test-jsonl data/marathi_v2_test_lines/labels.jsonl \
    --test-dir data/marathi_v2_test_lines \
    --model-path out/cp2_sft_merged \
    --max-new-tokens 320 --max-pixels 300000 \
    --out out/preds_lines.jsonl

# CLEAN CROPS (vs 66.89 zero-shot / 83.44 D)
python scripts/eval_chitrapathak.py \
    --test-jsonl data/marathi_v2_test_det_clean_fixed/labels.jsonl \
    --test-dir data/marathi_v2_test_det_clean \
    --model-path out/cp2_sft_merged \
    --out out/preds_clean.jsonl

# SHORT CROPS (vs 52.00 zero-shot / 73.33 D)
python scripts/eval_chitrapathak.py \
    --test-jsonl data/marathi_v3_short_test_det_fixed/labels.jsonl \
    --test-dir data/marathi_v3_short_test_det \
    --model-path out/cp2_sft_merged \
    --out out/preds_short.jsonl
```

## 7. Bring results home

```bash
scp root@<POD_IP>:/workspace/out/cp2_sft_lora/*.safetensors .   # adapter ~240MB
scp root@<POD_IP>:/workspace/out/preds_*.jsonl .                  # predictions
```
(No need to download the 7GB merged model — the adapter + base reproduces it.)

## 8. Decision criteria

- **Line characc > 96.5 and line exact ≥ 40%** → SFT worked; this becomes
  the production candidate (line-seg → VLM pipeline).
- **Fragment crops must not collapse**: clean ≥ ~70%, short ≥ ~55% would
  already beat zero-shot; D stays the fragment fallback either way.
- If space-normalization failures persist (check `preds_lines.jsonl` for
  `क्र. पदनि`-style reads), round 2: add line-level synthetic (render full
  lines from the v6/v7 vocab) and/or upweight lines ×16.

## Gotchas

- Python 3.11/3.12 on the pod image is fine; our local 3.14 venv was only
  for smoke-testing the plumbing.
- `transformers==5.16.1` pinned to match the baseline eval exactly.
- If `snapshot_download` is slow unauthenticated, set `HF_TOKEN`.
- The merged model dir also contains the processor config (max_pixels=300k)
  — keep them together.

---

# Part 2 — Gujarati + Joint experiment (2026-09-08)

## Background (measured locally)

- Chitrapathak-2 was NEVER trained on Gujarati (paper: 10 Indic languages,
  no Gujarati). Zero-shot on our Gujarati eval: **19.06 characc on 10 lines,
  0.91% exact on 110 crops** — it decodes the semantics but emits
  Devanagari/Punjabi script (script confusion). The Marathi adapter does not
  change this (18.50). SFT should fix the script mapping (same class as the
  digit-script issue that Marathi SFT fixed completely).
- Gujarati data is thin: 1,317 word crops + 40 lines (vs Marathi 8,886 + 130).
  In the joint mix Gujarati is upweighted (crops x3, lines x8 -> 30% share).

## The two runs (same pod session as Part 1 is fine)

### Run A — Gujarati-only adapter (~15 min)

```bash
python scripts/sft_chitrapathak.py \
    --model data/chitrapathak2 \
    --data data/cp2_sft/gujarati_only.jsonl \
    --out out/guj_sft_lora \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 1e-4 \
    --device cuda
```
(3 epochs — the set is small; watch the loss goes < 0.1)

### Run B — Joint Marathi+Gujarati adapter (~2h)

```bash
python scripts/sft_chitrapathak.py \
    --model data/chitrapathak2 \
    --data data/cp2_sft/joint_train.jsonl \
    --out out/joint_sft_lora \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 \
    --device cuda
```

### Merge both (as in Part 1)

```bash
python scripts/sft_chitrapathak.py --merge-only --model data/chitrapathak2 \
    --adapter out/guj_sft_lora --out out/guj_sft_merged
python scripts/sft_chitrapathak.py --merge-only --model data/chitrapathak2 \
    --adapter out/joint_sft_lora --out out/joint_sft_merged
```

## Evaluate BOTH adapters on BOTH languages

```bash
# Gujarati lines (10) — use --min-pixels 200000 (matches the zero-shot baseline)
python scripts/eval_chitrapathak.py \
    --test-jsonl data/gujarati_test_lines/labels.jsonl \
    --test-dir data/gujarati_test_lines --model-path out/guj_sft_merged \
    --min-pixels 200000 --max-new-tokens 320 --out out/guj_L_gujsonly.jsonl
python scripts/eval_chitrapathak.py \
    --test-jsonl data/gujarati_test_lines/labels.jsonl \
    --test-dir data/gujarati_test_lines --model-path out/joint_sft_merged \
    --min-pixels 200000 --max-new-tokens 320 --out out/guj_L_joint.jsonl

# Gujarati crops (110)
python scripts/eval_chitrapathak.py \
    --test-jsonl data/gujarati_test/labels.jsonl \
    --test-dir data/gujarati_test --model-path out/guj_sft_merged \
    --out out/guj_C_gujsonly.jsonl
python scripts/eval_chitrapathak.py \
    --test-jsonl data/gujarati_test/labels.jsonl \
    --test-dir data/gujarati_test --model-path out/joint_sft_merged \
    --out out/guj_C_joint.jsonl

# Marathi regression check (joint adapter must not lose Marathi)
# — the 90-line headline eval + 302 clean crops, same commands as Part 1
#   with --model-path out/joint_sft_merged
```

## Decision matrix

| Condition | Ship |
|---|---|
| Joint >= Gujarati-only on Gujarati AND Joint >= 93% of Marathi-only scores on Marathi | **one joint model** |
| Joint loses Marathi (>1pp characc drop on 90 lines) | two adapters, language-routed |
| Gujarati-only >= Joint on Gujarati by a lot | two adapters (joint diluted Gujarati) |

Baselines to compare: Gujarati zero-shot 19.06 characc / 0.91% crops;
Marathi SFT-only 98.86 characc / 51.1% lines / 81.79% clean crops.

Caveats: the 10-line Gujarati eval is noisy (1 line = 10pp) — trust the
110-crop set for granularity; if Gujarati lands < 90% characc, round 2 adds
the Gujarati synthetic (134MB, data/gujarati_synthetic) and/or asks the team
for a bigger batch.
