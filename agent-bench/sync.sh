#!/usr/bin/env bash
# Copy results off the rented machine with rclone:
#   RCLONE_DEST=remote:bucket/agent-bench agent-bench/sync.sh once|loop
# loop copies every SYNC_EVERY seconds (default 300), so a lost host costs at
# most a few minutes.  Configure the remote with a write-only key, since the
# host is not ours.  Live request logs are copied as they grow.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ -n "${RCLONE_DEST:-}" ] || die "set RCLONE_DEST, e.g. r2:bench/agent-bench"
command -v rclone >/dev/null || die "rclone not installed"
once() { rclone copy "$RESULTS_DIR" "$RCLONE_DEST/$(hostname)" --exclude '*.tmp' --transfers 8 -q; }

case ${1:-once} in
    once) once ;;
    loop) while true; do once || echo "sync: failed, retrying" >&2; sleep "${SYNC_EVERY:-300}"; done ;;
    *) die "usage: sync.sh once|loop" ;;
esac
