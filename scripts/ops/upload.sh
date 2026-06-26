#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PhysX-Mobility upload script (no-tar version)
#
# Goal: safe, resumable upload from a slow/fragile exFAT USB disk to TOS.
# Strategy:
#   - No tar. Use `tosutil cp -r -u` to sync each directory.
#   - Single-threaded I/O (-j 1 -p 1) so the USB bus is never saturated.
#   - Per-subdir for renders/ and reconstruction/ so a failure resumes cleanly.
#   - Optional dirty-page cap to prevent kernel memory pressure freezes.
#   - systemd-inhibit blocks sleep/shutdown but allows screen blanking.
# ============================================================================

SRC=${SRC:-/media/mi/QT16T/data/PhysX-Mobility}
TOS_DEST=${TOS_DEST:-tos://robot-data-lab/open-data/PhysX-Mobility-sjtu}

# State dir for lock + tosutil checkpoint. Kept on system disk (small files only).
STATE_DIR=${STATE_DIR:-/var/tmp/PhysX-Mobility_upload_state}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$STATE_DIR/.tos_checkpoint"}
LOCK_FILE=${LOCK_FILE:-"$STATE_DIR/upload.lock"}

# tosutil tuning. Keep these LOW. We are optimizing for "doesn't freeze",
# not for throughput. Increase only if you've confirmed the USB disk is healthy.
TOS_JOBS=${TOS_JOBS:-1}             # number of files in parallel
TOS_PARALLELS=${TOS_PARALLELS:-1}   # parts per file in parallel
TOS_THRESHOLD=${TOS_THRESHOLD:-104857600}   # 100 MiB: multipart cutoff
TOS_PART_SIZE=${TOS_PART_SIZE:-67108864}    # 64 MiB per part (small, gentler on USB)
TOS_VERIFY_CHECKSUM=${TOS_VERIFY_CHECKSUM:-1}

# Behavior flags
UPLOAD_INHIBIT_SLEEP=${UPLOAD_INHIBIT_SLEEP:-1}
REFUSE_DIRTY_EXFAT=${REFUSE_DIRTY_EXFAT:-1}
BETWEEN_DIR_SLEEP=${BETWEEN_DIR_SLEEP:-5}   # let USB cool down between dirs

# Optional kernel dirty-page cap. Requires sudo. Set to 0 to skip.
# These small values force the kernel to flush often instead of buffering
# gigabytes in RAM, which is what causes the system to lock up.
LIMIT_DIRTY_CACHE=${LIMIT_DIRTY_CACHE:-1}
DIRTY_BYTES=${DIRTY_BYTES:-$((64 * 1024 * 1024))}              # 64 MiB
DIRTY_BACKGROUND_BYTES=${DIRTY_BACKGROUND_BYTES:-$((32 * 1024 * 1024))}  # 32 MiB

# Top-level directories to upload as whole trees.
# NOTE: vlm/manifests/joint_transforms/preview/raw are already uploaded as
# .tar files on TOS from a previous run. Leaving this empty so we don't
# re-upload them. If you ever need to upload one of them as a directory tree,
# add it here (e.g. TOP_DIRS=(raw)).
TOP_DIRS=()

# Directories whose immediate children should each be uploaded as a separate
# sync job. This makes resume granular: if renders/objectX fails, only
# objectX restarts, not the whole renders/.
SPLIT_DIRS=(renders reconstruction)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

check_exfat_source_health() {
    local source_device fs_type
    source_device=$(findmnt -n -T "$SRC" -o SOURCE 2>/dev/null || true)
    fs_type=$(findmnt -n -T "$SRC" -o FSTYPE 2>/dev/null || true)

    if [[ "$fs_type" == "exfat" ]]; then
        log "Source is on exFAT ($source_device)."
        log "If the previous run hard-froze the machine, run 'sudo fsck.exfat $source_device' BEFORE retrying."

        if [[ "$REFUSE_DIRTY_EXFAT" == "1" ]] &&
            journalctl -b -k --no-pager 2>/dev/null |
                grep -F "exFAT-fs (${source_device##*/}): Volume was not properly unmounted" >/dev/null; then
            echo "Refusing to continue: current boot logged an unclean unmount on $source_device." >&2
            echo "Unmount it, run fsck.exfat, then retry. Override at your own risk:" >&2
            echo "    REFUSE_DIRTY_EXFAT=0 bash upload.sh" >&2
            exit 2
        fi
    fi
}

maybe_inhibit_sleep() {
    [[ "$UPLOAD_INHIBIT_SLEEP" == "1" ]] || return 0
    [[ "${UPLOAD_INHIBITED:-0}" == "1" ]] && return 0
    command -v systemd-inhibit >/dev/null 2>&1 || return 0

    local script_path
    script_path=$(readlink -f "${BASH_SOURCE[0]}")
    log "Re-execing under systemd-inhibit (blocks sleep/shutdown, allows screen blank)."
    exec systemd-inhibit \
        --what=sleep:shutdown \
        --who=PhysX-Mobility-upload \
        --why="Uploading PhysX-Mobility dataset" \
        --mode=block \
        env UPLOAD_INHIBITED=1 bash "$script_path" "$@"
}

maybe_limit_dirty_cache() {
    [[ "$LIMIT_DIRTY_CACHE" == "1" ]] || return 0

    # Try without sudo first (won't work for non-root, but harmless).
    if ! sudo -n true 2>/dev/null; then
        log "Skip dirty-cache cap: passwordless sudo not available."
        log "Recommended (run manually before starting):"
        log "  echo $DIRTY_BYTES            | sudo tee /proc/sys/vm/dirty_bytes"
        log "  echo $DIRTY_BACKGROUND_BYTES | sudo tee /proc/sys/vm/dirty_background_bytes"
        return 0
    fi

    log "Capping kernel dirty cache: dirty_bytes=$DIRTY_BYTES, dirty_background_bytes=$DIRTY_BACKGROUND_BYTES"
    echo "$DIRTY_BACKGROUND_BYTES" | sudo tee /proc/sys/vm/dirty_background_bytes >/dev/null
    echo "$DIRTY_BYTES"            | sudo tee /proc/sys/vm/dirty_bytes            >/dev/null
}

# tosutil cp -r -u <local_dir> <remote_dir>
# -u: skip files that already exist remotely with same size/checksum
# This gives us idempotent resume without any pre-listing.
sync_dir() {
    local local_dir=$1
    local remote_dir=$2
    local checksum_args=()

    if [[ "$TOS_VERIFY_CHECKSUM" == "1" ]]; then
        checksum_args=(-vchecksum)
    fi

    if [[ ! -d "$local_dir" ]]; then
        log "Skip missing local directory: $local_dir"
        return 0
    fi

    log "Syncing $local_dir  ->  $remote_dir"
    tosutil cp "$local_dir" "$remote_dir" \
        -r \
        -u \
        "${checksum_args[@]}" \
        -j "$TOS_JOBS" \
        -p "$TOS_PARALLELS" \
        -threshold "$TOS_THRESHOLD" \
        -ps "$TOS_PART_SIZE" \
        -cpd "$CHECKPOINT_DIR"

    if [[ "$BETWEEN_DIR_SLEEP" != "0" ]]; then
        sleep "$BETWEEN_DIR_SLEEP"
    fi
}

sync_split_dir() {
    local rel=$1
    local parent="$SRC/$rel"
    local dest_prefix="$TOS_DEST/$rel"
    local subdirs=() sub name

    if [[ ! -d "$parent" ]]; then
        log "Skip missing split directory: $parent"
        return 0
    fi

    echo "========================================="
    log "Split directory: $rel  (one upload per subdir)"
    echo "========================================="

    shopt -s nullglob
    subdirs=("$parent"/*/)
    if (( ${#subdirs[@]} == 0 )); then
        log "No subdirectories under $parent; syncing as whole dir."
        sync_dir "$parent" "$dest_prefix/"
        return 0
    fi

    for sub in "${subdirs[@]}"; do
        # Trailing slash stripped, then basename.
        name=$(basename "${sub%/}")
        echo "-----------------------------------------"
        log "[$rel] subdir: $name"
        echo "-----------------------------------------"
        # tosutil cp -r SRC DEST treats SRC's last component as a subdir to
        # create under DEST. If DEST=${dest_prefix}/${name}/, tosutil places
        # things at ${dest_prefix}/${name}/${name}/... (the bug that produced
        # renders/100013/100013/ on TOS). Passing DEST=${dest_prefix}/ lets
        # tosutil add exactly one ${name} level -> renders/100013/...
        sync_dir "$parent/$name" "$dest_prefix/"
    done
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

main() {
    maybe_inhibit_sleep "$@"

    require_cmd tosutil
    require_cmd flock
    require_cmd findmnt

    mkdir -p "$STATE_DIR" "$CHECKPOINT_DIR"

    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "Another upload.sh instance is already running. Lock: $LOCK_FILE" >&2
        exit 1
    fi

    log "Source:       $SRC"
    log "Destination:  $TOS_DEST"
    log "State dir:    $STATE_DIR"
    log "tosutil:      -j $TOS_JOBS -p $TOS_PARALLELS -threshold $TOS_THRESHOLD -ps $TOS_PART_SIZE -vchecksum $TOS_VERIFY_CHECKSUM"
    log "Inhibit:      sleep/shutdown blocked, screen blank allowed ($UPLOAD_INHIBIT_SLEEP)"
    df -h "$SRC" 2>/dev/null || true

    check_exfat_source_health
    maybe_limit_dirty_cache

    # Top-level whole-tree syncs.
    if (( ${#TOP_DIRS[@]} == 0 )); then
        log "TOP_DIRS is empty, skipping top-level directory uploads."
    else
        for d in "${TOP_DIRS[@]}"; do
            echo "========================================="
            log "Top-level dir: $d"
            echo "========================================="
            sync_dir "$SRC/$d" "$TOS_DEST/$d/"
        done
    fi

    # Per-subdir syncs (renders, reconstruction).
    for d in "${SPLIT_DIRS[@]}"; do
        sync_split_dir "$d"
    done

    echo "========================================="
    log "ALL DONE!"
    echo "========================================="
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
