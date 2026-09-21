#!/bin/bash
# ============================================================================
# Part 2 RECOVERY — run the remaining evals WITHOUT merging (disk-quota safe).
# Both adapters are already trained and saved:
#   out/guj_sft_lora, out/joint_sft_lora (and out/cp2_sft_lora from Part 1)
# This script frees disk, then evaluates adapters directly via --adapter.
# Marathi line evals use --max-pixels 300000 to match the official Part 1
# baseline protocol (98.86 characc / 51.11% exact).
# ============================================================================
set -e
cd "$(dirname "$0")"
mkdir -p out logs

banner() { echo; echo "=============== $* ==============="; }

# ---------------------------------------------------------------------------
banner "DISK CLEANUP (frees ~15GB+)"
rm -rf out/guj_sft_merged out/joint_sft_merged out/cp2_sft_merged
rm -rf ~/.cache/huggingface/hub/models--krutrim-ai-labs--Chitrapathak* 2>/dev/null || true
df -h /workspace | tail -1

# ---------------------------------------------------------------------------
# EVAL helper — adapter-based, no merged model on disk
ev() {  # ev <label> <adapter> <test_jsonl> <test_dir> [extra eval args...]
  local label=$1 adapter=$2 jsonl=$3 dir=$4; shift 4
  echo "--- eval: $label ---"
  python -u scripts/eval_chitrapathak.py \
      --test-jsonl "$jsonl" --test-dir "$dir" \
      --model-path data/chitrapathak2 --adapter "$adapter" \
      --out "out/preds_${label}.jsonl" "$@" 2>&1 | tee "logs/eval_${label}.log" | tail -4
}

banner "EVALS: Gujarati (both adapters)"
ev gujL_gujsonly out/guj_sft_lora data/gujarati_test_lines/labels.jsonl \
   data/gujarati_test_lines --min-pixels 200000 --max-new-tokens 320
ev gujL_joint out/joint_sft_lora data/gujarati_test_lines/labels.jsonl \
   data/gujarati_test_lines --min-pixels 200000 --max-new-tokens 320
ev gujC_gujsonly out/guj_sft_lora data/gujarati_test/labels.jsonl data/gujarati_test
ev gujC_joint out/joint_sft_lora data/gujarati_test/labels.jsonl data/gujarati_test

banner "EVALS: Marathi — joint adapter regression check (official protocol)"
ev marL_joint out/joint_sft_lora data/marathi_v2_test_lines/labels.jsonl \
   data/marathi_v2_test_lines --max-new-tokens 320 --max-pixels 300000
ev marC_joint out/joint_sft_lora data/marathi_v2_test_det_clean_fixed/labels.jsonl \
   data/marathi_v2_test_det_clean
ev marS_joint out/joint_sft_lora data/marathi_v3_short_test_det_fixed/labels.jsonl \
   data/marathi_v3_short_test_det

banner "EVALS: Marathi-only reference (same protocol, via adapter)"
ev marL_maronly out/cp2_sft_lora data/marathi_v2_test_lines/labels.jsonl \
   data/marathi_v2_test_lines --max-new-tokens 320 --max-pixels 300000
ev marC_maronly out/cp2_sft_lora data/marathi_v2_test_det_clean_fixed/labels.jsonl \
   data/marathi_v2_test_det_clean
ev marS_maronly out/cp2_sft_lora data/marathi_v3_short_test_det_fixed/labels.jsonl \
   data/marathi_v3_short_test_det

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
  Marathi-only official: 98.86 characc / 51.1% lines / 81.8% clean / 70.7% short

BRING HOME (from your Mac):
  scp root@<POD_IP>:/workspace/out/guj_sft_lora/*.safetensors .
  scp root@<POD_IP>:/workspace/out/guj_sft_lora/adapter_config.json .
  scp root@<POD_IP>:/workspace/out/joint_sft_lora/*.safetensors .
  scp root@<POD_IP>:/workspace/out/joint_sft_lora/adapter_config.json .
  scp root@<POD_IP>:/workspace/out/preds_*.jsonl .
  scp root@<POD_IP>:/workspace/logs/train_*.log .
EOF

echo; echo "ALL DONE $(date)"
