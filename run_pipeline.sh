#!/usr/bin/env bash
# Usage: bash run_pipeline.sh --config <inference.yaml>
# End-to-end articulated-object reconstruction pipeline (D-21..D-25).
# Chains 5 thin pipeline scripts via the single inference YAML config.

set -euo pipefail

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG=$2; shift 2 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done
[[ -z "$CONFIG" ]] && { echo "usage: $0 --config <inference.yaml>"; exit 2; }

# Parse YAML via Python (no yq dependency).
read -r OBJ_ID IMG_DIR OUT_ROOT SS_CKPT SS_DEC_CKPT PART_SS_CKPT SLAT_CKPT SLAT_DEC_CKPT TOKENS MASK_TOKEN_LABELS TARGET_PART_NAMES TARGET_SLOTS NUM_STEPS THRESHOLD < <(
  python -c "
import json, yaml, sys
c = yaml.safe_load(open('$CONFIG'))
print(c['object_id'], c['images_dir'], c['output_root'],
      c['ckpts']['ss_flow'], c['ckpts']['ss_decoder'], c['ckpts']['part_ss_latent_flow'],
      c['ckpts']['slat_flow'], c['ckpts']['slat_decoder'],
      c.get('cond_tokens', c.get('tokens', '')),
      c['mask_token_labels'],
      json.dumps(c['target_part_names'], separators=(',', ':')),
      json.dumps(c['target_slots'], separators=(',', ':')),
      c.get('num_steps', 25),
      c.get('threshold', 0.0))"
)

RUN_ID="$(date +%Y%m%d-%H%M%S)-${OBJ_ID}"
OUT="${OUT_ROOT}/${RUN_ID}"
mkdir -p "$OUT"
echo "[run_pipeline] run_id=$RUN_ID  out=$OUT"

python pipeline/01_ss_flow_mv.py    --images "$IMG_DIR"/*.png --ckpt "$SS_CKPT"     --num_steps "$NUM_STEPS" --output_dir "$OUT"
python pipeline/01_ss_decode.py     --ss_latent "$OUT"/ss_latent.npz --ckpt "$SS_DEC_CKPT" --threshold "$THRESHOLD" --output_dir "$OUT"
python pipeline/02_part_flow.py     --ss-latent "$OUT"/ss_latent.npz --cond-tokens "$TOKENS" --mask-token-labels "$MASK_TOKEN_LABELS" --target-part-names "$TARGET_PART_NAMES" --target-slots "$TARGET_SLOTS" --ckpt "$PART_SS_CKPT" --ss-decoder-ckpt "$SS_DEC_CKPT" --num-steps "$NUM_STEPS" --decode-threshold "$THRESHOLD" --output-dir "$OUT"
python pipeline/03_slat_flow.py     --images "$IMG_DIR"/*.png --occupancy "$OUT"/occupancy.npz --ckpt "$SLAT_CKPT" --num_steps "$NUM_STEPS" --output_dir "$OUT"
python pipeline/03_final_decode.py  --slat "$OUT"/slat.pt --ckpt "$SLAT_DEC_CKPT" --formats mesh,gaussian --output_dir "$OUT"

echo "[run_pipeline] DONE -> $OUT/{mesh.obj,gaussians.ply}"
