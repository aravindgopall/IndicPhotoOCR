#!/bin/bash
# ============================================================================
# Part 2 — Gujarati-only + Joint (Marathi+Gujarati) SFT experiment
# Run on the GPU pod, from /workspace (package v2 extracted, base model at
# data/chitrapathak2, deps installed).
#
#   bash run_all.sh            (foreground)
#   nohup bash run_all.sh > logs/run_part2.log 2>&1 &    (background)
#
# Produces: out/guj_sft_lora, out/joint_sft_lora (adapters),
#           out/guj_sft_merged, out/joint_sft_merged (merged models),
#           out/preds_*.jsonl (all predictions), logs/*.log
# ============================================================================
set -e
cd "$(dirname "$0")"
mkdir -p out logs

banner() { echo; echo "=============== $* ==============="; }

# ---------------------------------------------------------------------------
banner "TRAIN A: Gujarati-only adapter (3 epochs, ~15 min)"
python -u scripts/sft_chitrapathak.py \
    --model data/chitrapathak2 \
    --data data/cp2_sft/gujarati_only.jsonl \
    --out out/guj_sft_lora \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 1e-4 \
    --device cuda 2>&1 | tee logs/train_guj_only.log

# ---------------------------------------------------------------------------
banner "TRAIN B: Joint Marathi+Gujarati adapter (2 epochs, ~2h)"
python -u scripts/sft_chitrapathak.py \
    --model data/chitrapathak2 \
    --data data/cp2_sft/joint_train.jsonl \
    --out out/joint_sft_lora \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 \
    --device cuda 2>&1 | tee logs/train_joint.log

# ---------------------------------------------------------------------------
banner "MERGE both adapters"
python scripts/sft_chitrapathak.py --merge-only --model data/chitrapathak2 \
    --adapter out/guj_sft_lora --out out/guj_sft_merged 2>&1 | tail -1
python scripts/sft_chitrapathak.py --merge-only --model data/chitrapathak2 \
    --adapter out/joint_sft_lora --out out/joint_sft_merged 2>&1 | tail -1

# ---------------------------------------------------------------------------
# EVAL helper
ev() {  # ev <label> <model> <test_jsonl> <test_dir> [extra eval args...]
  local label=$1 model=$2 jsonl=$3 dir=$4; shift 4
  echo "--- eval: $label ---"
  python -u scripts/eval_chitrapathak.py \
      --test-jsonl "$jsonl" --test-dir "$dir" --model-path "$model" \
      --out "out/preds_${label}.jsonl" "$@" 2>&1 | tee "logs/eval_${label}.log" | tail -4
}

banner "EVALS: Gujarati (both adapters)"
# lines: --min-pixels 200000 matches the zero-shot baseline protocol
ev gujL_gujsonly out/guj_sft_merged data/gujarati_test_lines/labels.jsonl \
   data/gujarati_test_lines --min-pixels 200000 --max-new-tokens 320
ev gujL_joint out/joint_sft_merged data/gujarati_test_lines/labels.jsonl \
   data/gujarati_test_lines --min-pixels 200000 --max-new-tokens 320
# crops
ev gujC_gujsonly out/guj_sft_merged data/gujarati_test/labels.jsonl data/gujarati_test
ev gujC_joint out/joint_sft_merged data/gujarati_test/labels.jsonl data/gujarati_test

banner "EVALS: Marathi — joint adapter regression check"
ev marL_joint out/joint_sft_merged data/marathi_v2_test_lines/labels.jsonl \
   data/marathi_v2_test_lines --max-new-tokens 320
ev marC_joint out/joint_sft_merged data/marathi_v2_test_det_clean_fixed/labels.jsonl \
   data/marathi_v2_test_det_clean
ev marS_joint out/joint_sft_merged data/marathi_v3_short_test_det_fixed/labels.jsonl \
   data/marathi_v3_short_test_det

# same-session Marathi-only reference if the Part 1 merge is still on the pod
if [ -d out/cp2_sft_merged ]; then
  banner "EVALS: Marathi-only reference (out/cp2_sft_merged)"
  ev marL_maronly out/cp2_sft_merged data/marathi_v2_test_lines/labels.jsonl \
     data/marathi_v2_test_lines --max-new-tokens 320
  ev marC_maronly out/cp2_sft_merged data/marathi_v2_test_det_clean_fixed/labels.jsonl \
     data/marathi_v2_test_det_clean
  ev marS_maronly out/cp2_sft_merged data/marathi_v3_short_test_det_fixed/labels.jsonl \
     data/marathi_v3_short_test_det
fi

# ---------------------------------------------------------------------------
banner "SUMMARY"
echo "label           | raw exact / characc           | specials"
echo "----------------+-------------------------------+--------"
for f in logs/eval_*.log; do
  label=$(basename "$f" .log | sed 's/eval_//')
  raw=$(grep '^raw' "$f" | head -1 | sed 's/raw *//')
  sp=$(grep 'specials' "$f" | head -1 | tr -d 'specials: ')
  printf "%-15s | %-29s | %s\n" "$label" "$raw" "$sp"
done

cat << 'EOF'

DECISION MATRIX:
  - Joint >= Gujarati-only on Gujarati AND Joint holds Marathi
    (within ~1pp characc of Marathi-only on marL)  -> ship ONE joint model
  - Joint drops Marathi                            -> two adapters, language-routed
  - Gujarati-only >> Joint on Gujarati             -> two adapters (joint diluted)

BASELINES:
  Gujarati zero-shot: 19.06 characc (lines), 0.91% exact (crops)
  Marathi-only (this recipe): 98.9 characc / ~50% lines / ~81% clean / ~73% short

BRING HOME (from your Mac):
  scp root@<POD_IP>:/workspace/out/guj_sft_lora/*.safetensors .
  scp root@<POD_IP>:/workspace/out/guj_sft_lora/adapter_config.json .
  scp root@<POD_IP>:/workspace/out/joint_sft_lora/*.safetensors .
  scp root@<POD_IP>:/workspace/out/joint_sft_lora/adapter_config.json .
  scp root@<POD_IP>:/workspace/out/preds_*.jsonl .
  scp root@<POD_IP>:/workspace/logs/train_*.log .
EOF

echo; echo "ALL DONE $(date)"
