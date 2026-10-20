#!/bin/sh
# Ternary Bonsai 2 27B downloader (fork-owned; upstream's download_model.sh is
# left untouched).  Fetches the pinned PQ2_0 GGUF from Hugging Face into
# gguf/ternary-bonsai-2-27b/ (the path ds4's Bonsai tests default to) and
# verifies its sha256.
#
#   ./download_bonsai2.sh [--vision] [--token TOKEN]
#
# --vision also fetches the Q8_0 mmproj for --vision.
#
# DS4_GGUF_DIR overrides the gguf/ output directory, like download_model.sh.
set -e

REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
REVISION="b072e1d3b35a0a630cece372c2127528e0994386"
FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf"
SHA256="3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
VISION_FILE="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
VISION_SHA256="6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
OUT_DIR="$OUT_DIR/ternary-bonsai-2-27b"
TOKEN=${HF_TOKEN:-}
VISION=0

while [ $# -gt 0 ]; do
    case "$1" in
        --token) TOKEN="$2"; shift 2 ;;
        --vision) VISION=1; shift ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"

fetch() {
    dst="$OUT_DIR/$1"
    part="$dst.part"
    url="https://huggingface.co/$REPO/resolve/$REVISION/$1"
    if [ ! -f "$dst" ]; then
        echo "Downloading $1 from $REPO@$REVISION"
        if [ -n "$TOKEN" ]; then
            curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
        else
            curl -fL --progress-meter -C - -o "$part" "$url"
        fi
        mv "$part" "$dst"
    fi
    echo "Verifying sha256 of $dst"
    if command -v shasum >/dev/null 2>&1; then
        got=$(shasum -a 256 "$dst" | cut -d' ' -f1)
    else
        got=$(sha256sum "$dst" | cut -d' ' -f1)
    fi
    if [ "$got" != "$2" ]; then
        echo "sha256 mismatch: got $got, want $2" >&2
        exit 1
    fi
}

fetch "$FILE" "$SHA256"
if [ "$VISION" = 1 ]; then
    fetch "$VISION_FILE" "$VISION_SHA256"
    echo "Ready: ./ds4 -m $OUT_DIR/$FILE --vision $OUT_DIR/$VISION_FILE"
else
    echo "Ready: ./ds4 -m $OUT_DIR/$FILE"
fi
