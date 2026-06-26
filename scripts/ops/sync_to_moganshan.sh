#!/usr/bin/env bash
# scp scripts/train and TRELLIS-arts/trellis/models/part_predictor to moganshan.
# Two sequential scp connections (with brief pause to avoid rate limit).

set -euo pipefail

REMOTE="${REMOTE:-lai@180.184.148.169}"
PORT="${PORT:-10322}"
REMOTE_ROOT="${REMOTE_ROOT:-/moganshan/afs_a/lai/ccc/arts-reconstruction}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/cfy/cfy/ccc/nip/base_line/arts-reconstruction}"

echo "[sync] $LOCAL_ROOT -> $REMOTE:$REMOTE_ROOT"
echo ""

echo "[1/2] scp scripts/train"
scp -P "$PORT" -r "$LOCAL_ROOT/scripts/train" "$REMOTE:$REMOTE_ROOT/scripts/"

echo ""
echo "[wait] 5s before next scp..."
sleep 5

echo ""
echo "[2/2] scp TRELLIS-arts/trellis/models/part_predictor"
scp -P "$PORT" -r "$LOCAL_ROOT/TRELLIS-arts/trellis/models/part_predictor" \
       "$REMOTE:$REMOTE_ROOT/TRELLIS-arts/trellis/models/"

echo ""
echo "[done] synced to $REMOTE:$REMOTE_ROOT"
